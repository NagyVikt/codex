use anyhow::Context;
use anyhow::bail;
use clap::ArgAction;
use clap::Parser;
use codex_utils_cli::CliConfigOverrides;
use codex_tui::paste_image_to_temp_png;
use serde_json::json;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;
use tempfile::NamedTempFile;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::time::timeout;

/// Run and coordinate multiple Codex workers.
///
/// Design goals:
/// - Deterministic UX: same inputs => same plan => same artifacts.
/// - Production-grade ergonomics: session load/save, worker files, retries, timeout, output dir.
/// - Operational visibility: optional streaming, artifact capture (stdout/stderr/last message), report.
///
/// Notes:
/// - Execution is intentionally sequential because prompt chaining relies on previous outputs.
/// - Parallelism is a separate product decision (needs explicit dependency graph).
#[derive(Debug, Parser)]
#[command(
    name = "commander",
    about = "Run and coordinate multiple Codex workers."
)]
pub struct CommanderCli {
    /// Shared task that every worker should contribute to.
    #[arg(
        long = "task",
        short = 't',
        value_name = "TASK",
        conflicts_with = "task_file"
    )]
    pub task: Option<String>,

    /// Read the shared task from a file (UTF-8).
    #[arg(long = "task-file", value_name = "PATH", conflicts_with = "task")]
    pub task_file: Option<PathBuf>,

    /// Worker definition in NAME=INSTRUCTIONS form.
    ///
    /// Example:
    /// `--worker planner="Create a concrete implementation plan"`
    #[arg(
        long = "worker",
        short = 'w',
        value_name = "NAME=INSTRUCTIONS",
        action = ArgAction::Append
    )]
    pub worker_specs: Vec<String>,

    /// Load workers from a file (one NAME=INSTRUCTIONS per line; '#' starts a comment).
    #[arg(long = "worker-file", value_name = "PATH", action = ArgAction::Append)]
    pub worker_files: Vec<PathBuf>,

    /// Model forwarded to each worker session.
    #[arg(long = "model", short = 'm')]
    pub model: Option<String>,

    /// Configuration profile from config.toml to use for each worker.
    #[arg(long = "profile", short = 'p')]
    pub config_profile: Option<String>,

    /// Forward `--full-auto` to each worker.
    #[arg(long = "full-auto", default_value_t = false)]
    pub full_auto: bool,

    /// Forward `--dangerously-bypass-approvals-and-sandbox` to each worker.
    #[arg(
        long = "dangerously-bypass-approvals-and-sandbox",
        alias = "yolo",
        default_value_t = false,
        conflicts_with = "full_auto"
    )]
    pub dangerously_bypass_approvals_and_sandbox: bool,

    /// Per-worker environment variables forwarded to Codex exec (KEY=VALUE).
    #[arg(long = "env", value_name = "KEY=VALUE", action = ArgAction::Append)]
    pub env: Vec<String>,

    /// Emit one JSON object per worker instead of formatted text (NDJSON).
    #[arg(long = "json", default_value_t = false)]
    pub json: bool,

    /// Keep an interactive commander prompt open.
    ///
    /// Supported slash commands:
    /// `/task`, `/add-worker`, `/remove-worker`, `/run`, `/workers`, `/tree`,
    /// `/results`, `/show`, `/config`, `/save`, `/load`, `/clear-results`,
    /// `/image`, `/attach-image`, `/image-paste`,
    /// `/set-model`, `/set-profile`, `/toggle-full-auto`, `/toggle-yolo`,
    /// `/set-timeout`, `/set-retries`, `/toggle-continue-on-failure`,
    /// `/toggle-auto-reviewer`, `/toggle-finalize`, `/set-output-dir`,
    /// `/toggle-stream`, `/toggle-dry-run`,
    /// `/help`, `/quit`.
    #[arg(long = "shell", default_value_t = false, conflicts_with = "json")]
    pub shell: bool,

    /// Don't auto-inject the default reviewer worker.
    #[arg(long = "no-auto-reviewer", default_value_t = false)]
    pub no_auto_reviewer: bool,

    /// Override the default auto-reviewer instructions (only used when auto-reviewer is enabled).
    #[arg(long = "reviewer-instructions", value_name = "TEXT")]
    pub reviewer_instructions: Option<String>,

    /// Append a final "synthesizer" worker to consolidate outputs into a single deliverable.
    #[arg(long = "finalize", default_value_t = false)]
    pub finalize: bool,

    /// Per-worker timeout, in seconds. If set, a worker process exceeding this deadline is killed.
    #[arg(long = "timeout-secs", value_name = "SECS")]
    pub timeout_secs: Option<u64>,

    /// Number of retries for each worker on failure (exit != 0 or timeout).
    #[arg(long = "retries", default_value_t = 0)]
    pub retries: usize,

    /// Keep running remaining workers even when one fails. Commander will still exit non-zero at end.
    #[arg(long = "continue-on-failure", default_value_t = false)]
    pub continue_on_failure: bool,

    /// Limit how many previous worker outputs are embedded into the next worker prompt.
    ///
    /// - Not set: include all previous outputs.
    /// - 0: include none.
    /// - N: include the last N outputs.
    #[arg(long = "previous-outputs", value_name = "N")]
    pub previous_outputs: Option<usize>,

    /// Hard cap on characters included from previous outputs in prompts (newest outputs win).
    #[arg(long = "max-context-chars", value_name = "N")]
    pub max_context_chars: Option<usize>,

    /// Directory to persist prompts + outputs + logs.
    ///
    /// Files are written as: `<dir>/<NNN>_<worker>_{prompt,last_message,stdout,stderr}.txt`
    #[arg(long = "output-dir", value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Save prompts alongside outputs (requires --output-dir).
    #[arg(
        long = "save-prompts",
        default_value_t = false,
        requires = "output_dir"
    )]
    pub save_prompts: bool,

    /// Write a consolidated markdown + JSON report in output-dir after execution (requires --output-dir).
    #[arg(
        long = "write-report",
        default_value_t = false,
        requires = "output_dir"
    )]
    pub write_report: bool,

    /// Stream worker stdout/stderr live with a worker prefix.
    #[arg(long = "stream", default_value_t = false, conflicts_with = "json")]
    pub stream: bool,

    /// Print prompts and execution plan, but do not execute Codex.
    #[arg(long = "dry-run", default_value_t = false)]
    pub dry_run: bool,

    /// Load task + workers from a session file (JSON).
    #[arg(long = "load-session", value_name = "PATH")]
    pub load_session: Option<PathBuf>,

    /// Save task + workers to a session file (JSON) after run (even if the run fails).
    #[arg(long = "save-session", value_name = "PATH")]
    pub save_session: Option<PathBuf>,
}

const DEFAULT_REVIEWER_INSTRUCTIONS: &str = "Review every worker output above. Point out problems, missing tests, regressions, and concrete fixes. Keep feedback actionable.";
const DEFAULT_SYNTHESIZER_INSTRUCTIONS: &str = "Consolidate all worker outputs into a single final deliverable. Resolve conflicts, pick the best decisions, and produce a crisp actionable result. If code changes are referenced, include a minimal patch plan and test plan.";

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkerDefinition {
    name: String,
    instructions: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AutoReviewerConfig {
    enabled: bool,
    instructions: String,
}

impl Default for AutoReviewerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            instructions: DEFAULT_REVIEWER_INSTRUCTIONS.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreviousOutputsSelection {
    All,
    None,
    Last(usize),
}

impl PreviousOutputsSelection {
    fn from_cli_flag(flag: Option<usize>) -> Self {
        match flag {
            None => Self::All,
            Some(0) => Self::None,
            Some(n) => Self::Last(n),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommanderRunConfig {
    model: Option<String>,
    config_profile: Option<String>,
    full_auto: bool,
    yolo: bool,
    env: Vec<(String, String)>,

    json: bool,
    stream: bool,
    dry_run: bool,

    auto_reviewer: AutoReviewerConfig,
    finalize: bool,

    timeout: Option<Duration>,
    retries: usize,
    continue_on_failure: bool,

    previous_outputs: PreviousOutputsSelection,
    max_context_chars: Option<usize>,

    output_dir: Option<PathBuf>,
    save_prompts: bool,
    write_report: bool,
}

impl Default for CommanderRunConfig {
    fn default() -> Self {
        Self {
            model: None,
            config_profile: None,
            full_auto: false,
            yolo: false,
            env: Vec::new(),
            json: false,
            stream: false,
            dry_run: false,
            auto_reviewer: AutoReviewerConfig::default(),
            finalize: false,
            timeout: None,
            retries: 0,
            continue_on_failure: false,
            previous_outputs: PreviousOutputsSelection::All,
            max_context_chars: None,
            output_dir: None,
            save_prompts: false,
            write_report: false,
        }
    }
}

impl CommanderRunConfig {
    fn from_cli(cli: &CommanderCli) -> anyhow::Result<Self> {
        let env = cli
            .env
            .iter()
            .map(|raw| parse_env_kv(raw))
            .collect::<anyhow::Result<Vec<_>>>()
            .context("invalid --env")?;

        let reviewer_instructions = cli
            .reviewer_instructions
            .clone()
            .unwrap_or_else(|| DEFAULT_REVIEWER_INSTRUCTIONS.to_string());

        Ok(Self {
            model: cli.model.clone(),
            config_profile: cli.config_profile.clone(),
            full_auto: cli.full_auto,
            yolo: cli.dangerously_bypass_approvals_and_sandbox,
            env,

            json: cli.json,
            stream: cli.stream,
            dry_run: cli.dry_run,

            auto_reviewer: AutoReviewerConfig {
                enabled: !cli.no_auto_reviewer,
                instructions: reviewer_instructions,
            },
            finalize: cli.finalize,

            timeout: cli.timeout_secs.map(Duration::from_secs),
            retries: cli.retries,
            continue_on_failure: cli.continue_on_failure,

            previous_outputs: PreviousOutputsSelection::from_cli_flag(cli.previous_outputs),
            max_context_chars: cli.max_context_chars,

            output_dir: cli.output_dir.clone(),
            save_prompts: cli.save_prompts,
            write_report: cli.write_report,
        })
    }

    fn enforce_flag_invariants(&mut self) {
        // full-auto and yolo are mutually exclusive. In shell mode, toggles can violate this.
        if self.full_auto && self.yolo {
            self.yolo = false;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    TimedOut,
    Skipped,
}

impl WorkerStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
            Self::Skipped => "skipped",
        }
    }

    fn is_done(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::TimedOut | Self::Skipped
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkerRuntimeState {
    name: String,
    status: WorkerStatus,
    output_chars: usize,
    attempts: usize,
    duration_ms: Option<u128>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkerExecutionArtifacts {
    last_message_path: PathBuf,
    prompt_path: Option<PathBuf>,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct CommanderSession {
    task: Option<String>,
    workers: Vec<WorkerDefinition>,
    worker_outputs: Vec<(String, String)>,
    worker_states: Vec<WorkerRuntimeState>,
    run_config: CommanderRunConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CommanderShellCommand {
    Task(String),
    AttachImage(Option<String>),
    PasteClipboardImage,
    AddWorker(String),
    RemoveWorker(String),
    Run,
    Workers,
    Tree,
    Results,
    Show,
    Config,
    Save(String),
    Load(String),
    ClearResults,
    SetModel(String),
    SetProfile(String),
    ToggleFullAuto,
    ToggleYolo,
    SetTimeout(Option<u64>),
    SetRetries(usize),
    ToggleContinueOnFailure,
    ToggleAutoReviewer,
    ToggleFinalize,
    SetOutputDir(Option<String>),
    ToggleStream,
    ToggleDryRun,
    Help,
    Quit,
    Empty,
    UsageError(String),
    Unknown(String),
}

pub async fn run(
    cli: CommanderCli,
    root_config_overrides: CliConfigOverrides,
) -> anyhow::Result<()> {
    let codex_bin = std::env::current_exe().context("failed to determine the codex binary path")?;

    let mut session = CommanderSession::default();
    session.run_config = CommanderRunConfig::from_cli(&cli)?;

    if let Some(path) = &cli.load_session {
        load_session_into(&mut session, path)?;
    }

    // CLI task overrides session task.
    if let Some(task) = &cli.task {
        session.task = Some(task.clone());
    }
    if let Some(task_file) = &cli.task_file {
        session.task = Some(read_task_from_file(task_file)?);
    }

    // Merge worker specs from CLI and worker files into session (upsert by name case-insensitively).
    for (index, raw) in cli.worker_specs.iter().enumerate() {
        let position = index + 1;
        let parsed = parse_worker_definition(raw)
            .with_context(|| format!("invalid --worker argument at position {position}: {raw}"))?;
        upsert_worker(&mut session.workers, parsed);
    }

    for worker_file in &cli.worker_files {
        let loaded = parse_worker_file(worker_file)
            .with_context(|| format!("failed to read --worker-file {}", worker_file.display()))?;
        for worker in loaded {
            upsert_worker(&mut session.workers, worker);
        }
    }

    refresh_session_worker_states(&mut session);

    let should_open_shell = cli.shell
        || (!cli.json
            && session
                .task
                .as_deref()
                .map(str::trim)
                .unwrap_or_default()
                .is_empty()
            && session.workers.is_empty());

    let result = if should_open_shell {
        run_commander_shell(&mut session, &root_config_overrides, &codex_bin).await
    } else {
        validate_session_for_run(&session)?;
        execute_workers(&mut session, &root_config_overrides, &codex_bin).await
    };

    // Persist session even if execution failed: session files are a workflow primitive.
    if let Some(path) = &cli.save_session {
        if let Err(err) = save_session(&session, path) {
            eprintln!("Failed to save session to {}: {err}", path.display());
        }
    }

    // Bubble up execution result.
    result
}

fn validate_session_for_run(session: &CommanderSession) -> anyhow::Result<()> {
    if session
        .task
        .as_deref()
        .map(|s| s.trim())
        .unwrap_or("")
        .is_empty()
    {
        bail!("no task is set. pass --task/--task-file or run with --shell and set one via /task");
    }
    if session.workers.is_empty() {
        bail!(
            "no workers are set. pass at least one --worker/--worker-file or run with --shell and add via /add-worker"
        );
    }

    // Enforce name uniqueness (case-insensitive). Upsert should already avoid this, but validate anyway.
    let mut seen = HashSet::new();
    for worker in &session.workers {
        let key = worker.name.to_ascii_lowercase();
        if !seen.insert(key) {
            bail!("duplicate worker name detected: '{}'", worker.name);
        }
    }

    if session.run_config.save_prompts && session.run_config.output_dir.is_none() {
        bail!("--save-prompts requires --output-dir");
    }
    if session.run_config.write_report && session.run_config.output_dir.is_none() {
        bail!("--write-report requires --output-dir");
    }

    Ok(())
}

fn worker_states_from_workers(workers: &[WorkerDefinition]) -> Vec<WorkerRuntimeState> {
    workers
        .iter()
        .map(|worker| WorkerRuntimeState {
            name: worker.name.clone(),
            status: WorkerStatus::Pending,
            output_chars: 0,
            attempts: 0,
            duration_ms: None,
        })
        .collect()
}

fn refresh_session_worker_states(session: &mut CommanderSession) {
    let execution_workers = execution_workers(session);
    session.worker_states = worker_states_from_workers(&execution_workers);
}

fn execution_workers(session: &CommanderSession) -> Vec<WorkerDefinition> {
    execution_workers_with_auto_workers(&session.workers, &session.run_config)
}

fn execution_workers_with_auto_workers(
    workers: &[WorkerDefinition],
    config: &CommanderRunConfig,
) -> Vec<WorkerDefinition> {
    let mut execution_workers = workers.to_vec();

    // Auto reviewer.
    if config.auto_reviewer.enabled
        && !workers.is_empty()
        && !workers
            .iter()
            .any(|worker| worker.name.eq_ignore_ascii_case("reviewer"))
    {
        execution_workers.push(WorkerDefinition {
            name: "reviewer".to_string(),
            instructions: config.auto_reviewer.instructions.clone(),
        });
    }

    // Optional synthesizer/finalizer.
    if config.finalize
        && !workers.is_empty()
        && !execution_workers
            .iter()
            .any(|worker| worker.name.eq_ignore_ascii_case("synthesizer"))
    {
        execution_workers.push(WorkerDefinition {
            name: "synthesizer".to_string(),
            instructions: DEFAULT_SYNTHESIZER_INSTRUCTIONS.to_string(),
        });
    }

    execution_workers
}

async fn execute_workers(
    session: &mut CommanderSession,
    root_config_overrides: &CliConfigOverrides,
    codex_bin: &Path,
) -> anyhow::Result<()> {
    validate_session_for_run(session)?;

    session.run_config.enforce_flag_invariants();

    let task = session
        .task
        .clone()
        .context("task should exist after validation")?;

    session.worker_outputs.clear();

    let execution_workers = execution_workers(session);
    session.worker_states = worker_states_from_workers(&execution_workers);

    if !session.run_config.json {
        print_worker_statuses(&session.worker_states);
    }

    if let Some(output_dir) = &session.run_config.output_dir {
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("failed to create output-dir at {}", output_dir.display()))?;
    }

    let mut failures: Vec<String> = Vec::new();

    for (index, worker) in execution_workers.into_iter().enumerate() {
        session.worker_states[index].status = WorkerStatus::Running;
        if !session.run_config.json {
            print_worker_statuses(&session.worker_states);
        }

        let prompt =
            build_worker_prompt(&task, &worker, &session.worker_outputs, &session.run_config);

        if session.run_config.dry_run {
            session.worker_states[index].status = WorkerStatus::Skipped;
            session.worker_states[index].attempts = 0;
            session.worker_states[index].duration_ms = Some(0);
            if session.run_config.json {
                let payload = json!({
                    "worker": worker.name,
                    "status": session.worker_states[index].status.label(),
                    "prompt": prompt,
                });
                println!("{payload}");
            } else {
                println!("[{}] (dry-run)\n{}\n", worker.name, prompt);
                print_worker_statuses(&session.worker_states);
            }
            continue;
        }

        let worker_name = worker.name.clone();
        let start = Instant::now();

        let artifacts = prepare_worker_artifacts(session, index, &worker_name)?;
        if session.run_config.save_prompts {
            if let Some(prompt_path) = &artifacts.prompt_path {
                std::fs::write(prompt_path, &prompt).with_context(|| {
                    format!("failed to write prompt file {}", prompt_path.display())
                })?;
            }
        }

        let mut attempt = 0usize;
        let mut last_error: Option<String> = None;
        let mut final_output: Option<String> = None;
        let mut final_stdout = String::new();
        let mut final_stderr = String::new();
        let mut final_status: Option<WorkerStatus> = None;

        while attempt <= session.run_config.retries {
            attempt += 1;

            let exec_result = execute_single_worker_attempt(
                &worker,
                &prompt,
                &artifacts,
                session,
                root_config_overrides,
                codex_bin,
            )
            .await;

            match exec_result {
                Ok(attempt_result) => {
                    final_output = Some(attempt_result.output.clone());
                    final_stdout = attempt_result.stdout.clone();
                    final_stderr = attempt_result.stderr.clone();

                    session
                        .worker_outputs
                        .push((worker.name.clone(), attempt_result.output.clone()));

                    let output_chars = attempt_result.output.chars().count();
                    session.worker_states[index].output_chars = output_chars;
                    session.worker_states[index].attempts = attempt;
                    session.worker_states[index].duration_ms = Some(start.elapsed().as_millis());
                    session.worker_states[index].status = WorkerStatus::Succeeded;
                    final_status = Some(WorkerStatus::Succeeded);

                    if session.run_config.json {
                        let payload = json!({
                            "worker": worker_name,
                            "status": session.worker_states[index].status.label(),
                            "attempts": attempt,
                            "duration_ms": session.worker_states[index].duration_ms,
                            "output": attempt_result.output,
                        });
                        println!("{payload}");
                    } else {
                        println!("[{worker_name}]\n{}\n", attempt_result.output);
                        print_worker_statuses(&session.worker_states);
                    }
                    break;
                }
                Err(err) => {
                    let elapsed = start.elapsed().as_millis();
                    let err_string = err.to_string();
                    last_error = Some(err_string.clone());

                    // Attempt result already wrote stdout/stderr to artifacts; pull them for printing.
                    // Best-effort read for operator visibility.
                    if let Ok(s) = std::fs::read_to_string(&artifacts.stdout_path) {
                        final_stdout = s;
                    }
                    if let Ok(s) = std::fs::read_to_string(&artifacts.stderr_path) {
                        final_stderr = s;
                    }

                    let timed_out = err_string.contains("timed out");

                    session.worker_states[index].attempts = attempt;
                    session.worker_states[index].duration_ms = Some(elapsed);
                    session.worker_states[index].status = if timed_out {
                        WorkerStatus::TimedOut
                    } else {
                        WorkerStatus::Failed
                    };
                    final_status = Some(session.worker_states[index].status);

                    let will_retry = attempt <= session.run_config.retries;
                    if session.run_config.json {
                        let payload = json!({
                            "worker": worker_name,
                            "status": session.worker_states[index].status.label(),
                            "attempts": attempt,
                            "duration_ms": elapsed,
                            "error": err_string,
                            "will_retry": will_retry,
                        });
                        println!("{payload}");
                    } else {
                        let status_label = session.worker_states[index].status.label();
                        eprintln!(
                            "[{worker_name}] {status_label} (attempt {attempt}/{}) after {elapsed}ms: {err_string}",
                            session.run_config.retries + 1
                        );
                        if will_retry {
                            eprintln!("[{worker_name}] retrying...");
                        }
                        print_worker_statuses(&session.worker_states);
                    }

                    if !will_retry {
                        break;
                    }

                    // Basic backoff: 250ms, 500ms, 1000ms...
                    let backoff_ms =
                        250u64.saturating_mul(2u64.saturating_pow((attempt - 1) as u32));
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                }
            }
        }

        let status = final_status.unwrap_or(WorkerStatus::Failed);
        if status != WorkerStatus::Succeeded {
            let message = format!(
                "worker '{}' {} after {} attempt(s). last error: {}",
                worker_name,
                status.label(),
                attempt,
                last_error.unwrap_or_else(|| "<unknown>".to_string())
            );
            failures.push(message);

            if !session.run_config.continue_on_failure {
                let stdout_trimmed = final_stdout.trim();
                let stderr_trimmed = final_stderr.trim();
                bail!(
                    "worker '{worker_name}' failed.\nstdout:\n{stdout_trimmed}\nstderr:\n{stderr_trimmed}"
                );
            }
        } else if let Some(output_dir) = &session.run_config.output_dir {
            // Write incremental report slices for succeeded workers (optional and cheap).
            // Main report is written once at end if enabled.
            let _ = write_worker_snapshot_md(
                output_dir,
                index + 1,
                &worker_name,
                final_output.as_deref().unwrap_or(""),
            );
        }
    }

    if let Some(output_dir) = &session.run_config.output_dir {
        if session.run_config.write_report {
            write_reports(output_dir, session)?;
        }
    }

    if !failures.is_empty() {
        let mut msg = String::from("one or more workers failed:\n");
        for f in failures {
            msg.push_str("- ");
            msg.push_str(&f);
            msg.push('\n');
        }
        bail!(msg);
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct AttemptResult {
    output: String,
    stdout: String,
    stderr: String,
}

async fn execute_single_worker_attempt(
    worker: &WorkerDefinition,
    prompt: &str,
    artifacts: &WorkerExecutionArtifacts,
    session: &CommanderSession,
    root_config_overrides: &CliConfigOverrides,
    codex_bin: &Path,
) -> anyhow::Result<AttemptResult> {
    let mut command = tokio::process::Command::new(codex_bin);
    command
        .arg("exec")
        .arg("--skip-git-repo-check")
        .arg("--output-last-message")
        .arg(&artifacts.last_message_path)
        .arg(prompt)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Run config passthrough
    if let Some(model) = &session.run_config.model {
        command.arg("--model").arg(model);
    }
    if let Some(config_profile) = &session.run_config.config_profile {
        command.arg("--profile").arg(config_profile);
    }
    if session.run_config.full_auto {
        command.arg("--full-auto");
    }
    if session.run_config.yolo {
        command.arg("--dangerously-bypass-approvals-and-sandbox");
    }
    for raw_override in &root_config_overrides.raw_overrides {
        command.arg("--config").arg(raw_override);
    }
    for (k, v) in &session.run_config.env {
        command.env(k, v);
    }

    // Execute with streaming + timeout.
    let worker_name = &worker.name;
    let timeout = session.run_config.timeout;

    let run = run_child_with_optional_stream(command, worker_name, session.run_config.stream);

    let run_result = if let Some(timeout) = timeout {
        match tokio::time::timeout(timeout, run).await {
            Ok(res) => res,
            Err(_) => {
                // Timeout already occurred; best-effort write markers.
                let marker = format!("timed out after {}s\n", timeout.as_secs());
                let _ = std::fs::write(&artifacts.stderr_path, &marker);
                bail!(
                    "worker '{worker_name}' timed out after {}s",
                    timeout.as_secs()
                );
            }
        }
    } else {
        run.await
    }?;

    // Persist stdout/stderr as artifacts (always).
    std::fs::write(&artifacts.stdout_path, &run_result.stdout).with_context(|| {
        format!(
            "failed to write stdout log {}",
            artifacts.stdout_path.display()
        )
    })?;
    std::fs::write(&artifacts.stderr_path, &run_result.stderr).with_context(|| {
        format!(
            "failed to write stderr log {}",
            artifacts.stderr_path.display()
        )
    })?;

    if !run_result.status_success {
        let status = run_result.status_label;
        let stdout_trimmed = run_result.stdout.trim();
        let stderr_trimmed = run_result.stderr.trim();
        bail!(
            "worker '{worker_name}' failed with status {status}.\nstdout:\n{stdout_trimmed}\nstderr:\n{stderr_trimmed}"
        );
    }

    // Primary output comes from the last-message file; fallback to stdout if empty.
    let mut worker_output =
        std::fs::read_to_string(&artifacts.last_message_path).with_context(|| {
            format!(
                "worker '{worker_name}' completed but last message file was not readable: {}",
                artifacts.last_message_path.display()
            )
        })?;

    if worker_output.trim().is_empty() {
        worker_output = run_result.stdout.clone();
    }

    let worker_output = worker_output.trim().to_string();
    Ok(AttemptResult {
        output: worker_output,
        stdout: run_result.stdout,
        stderr: run_result.stderr,
    })
}

#[derive(Debug, Clone)]
struct ChildRunResult {
    status_success: bool,
    status_label: String,
    stdout: String,
    stderr: String,
}

async fn run_child_with_optional_stream(
    mut command: tokio::process::Command,
    worker_name: &str,
    stream: bool,
) -> anyhow::Result<ChildRunResult> {
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn worker '{worker_name}'"))?;

    let stdout = child
        .stdout
        .take()
        .context("child stdout pipe should exist")?;
    let stderr = child
        .stderr
        .take()
        .context("child stderr pipe should exist")?;

    let stdout_task = tokio::spawn(read_stream_lines(
        stdout,
        worker_name.to_string(),
        "stdout",
        stream,
    ));
    let stderr_task = tokio::spawn(read_stream_lines(
        stderr,
        worker_name.to_string(),
        "stderr",
        stream,
    ));

    let status = child
        .wait()
        .await
        .with_context(|| format!("failed waiting for worker '{worker_name}'"))?;

    let stdout = stdout_task.await.context("stdout task join failed")??;
    let stderr = stderr_task.await.context("stderr task join failed")??;

    let status_success = status.success();
    let status_label = format!("{status}");

    Ok(ChildRunResult {
        status_success,
        status_label,
        stdout,
        stderr,
    })
}

async fn read_stream_lines<R: tokio::io::AsyncRead + Unpin>(
    reader: R,
    worker: String,
    stream_name: &'static str,
    stream: bool,
) -> anyhow::Result<String> {
    let mut out = String::new();
    let mut lines = BufReader::new(reader).lines();
    while let Some(line) = lines.next_line().await? {
        if stream {
            // Keep it prefix-stable for downstream log processing.
            // Example: [planner:stdout] ...
            println!("[{worker}:{stream_name}] {line}");
        }
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

fn prepare_worker_artifacts(
    session: &CommanderSession,
    index: usize,
    worker_name: &str,
) -> anyhow::Result<WorkerExecutionArtifacts> {
    if let Some(output_dir) = &session.run_config.output_dir {
        let base = worker_artifact_base_name(index + 1, worker_name);
        let last_message_path = output_dir.join(format!("{base}_last_message.txt"));
        let stdout_path = output_dir.join(format!("{base}_stdout.txt"));
        let stderr_path = output_dir.join(format!("{base}_stderr.txt"));
        let prompt_path = if session.run_config.save_prompts {
            Some(output_dir.join(format!("{base}_prompt.txt")))
        } else {
            None
        };

        Ok(WorkerExecutionArtifacts {
            last_message_path,
            prompt_path,
            stdout_path,
            stderr_path,
        })
    } else {
        // Temp files per worker attempt (last message); stdout/stderr are also temp (in-memory)
        // but we still materialize them to stable temp files for consistent debug in retry loops.
        let output_file = NamedTempFile::new().context("failed to create worker output file")?;
        let stdout_file = NamedTempFile::new().context("failed to create worker stdout file")?;
        let stderr_file = NamedTempFile::new().context("failed to create worker stderr file")?;

        Ok(WorkerExecutionArtifacts {
            last_message_path: output_file.path().to_path_buf(),
            prompt_path: None,
            stdout_path: stdout_file.path().to_path_buf(),
            stderr_path: stderr_file.path().to_path_buf(),
        })
    }
}

fn worker_artifact_base_name(index_1: usize, worker_name: &str) -> String {
    let safe = sanitize_filename_component(worker_name);
    format!("{index_1:03}_{safe}")
}

fn sanitize_filename_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "worker".to_string()
    } else {
        out
    }
}

fn upsert_worker(workers: &mut Vec<WorkerDefinition>, worker: WorkerDefinition) {
    if let Some(existing) = workers
        .iter_mut()
        .find(|w| w.name.eq_ignore_ascii_case(&worker.name))
    {
        existing.name = worker.name;
        existing.instructions = worker.instructions;
    } else {
        workers.push(worker);
    }
}

fn read_task_from_file(path: &Path) -> anyhow::Result<String> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read task file {}", path.display()))?;
    let task = raw.trim().to_string();
    if task.is_empty() {
        bail!("task file {} is empty", path.display());
    }
    Ok(task)
}

fn parse_worker_file(path: &Path) -> anyhow::Result<Vec<WorkerDefinition>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read worker file {}", path.display()))?;

    let mut workers = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parsed = parse_worker_definition(trimmed).with_context(|| {
            format!(
                "invalid worker spec in {} at line {}: {}",
                path.display(),
                idx + 1,
                trimmed
            )
        })?;
        upsert_worker(&mut workers, parsed);
    }

    Ok(workers)
}

fn parse_env_kv(raw: &str) -> anyhow::Result<(String, String)> {
    let Some((k, v)) = raw.split_once('=') else {
        bail!("expected KEY=VALUE");
    };
    let k = k.trim();
    let v = v.trim();
    if k.is_empty() {
        bail!("env var key cannot be empty");
    }
    // Allow empty values: KEY=
    Ok((k.to_string(), v.to_string()))
}

fn parse_worker_definition(spec: &str) -> anyhow::Result<WorkerDefinition> {
    let Some((name, instructions)) = spec.split_once('=') else {
        bail!("expected NAME=INSTRUCTIONS");
    };
    let name = name.trim();
    let instructions = instructions.trim();

    if name.is_empty() {
        bail!("worker name cannot be empty");
    }
    if instructions.is_empty() {
        bail!("worker instructions cannot be empty");
    }

    Ok(WorkerDefinition {
        name: name.to_string(),
        instructions: instructions.to_string(),
    })
}

fn build_worker_prompt(
    task: &str,
    worker: &WorkerDefinition,
    previous_outputs: &[(String, String)],
    config: &CommanderRunConfig,
) -> String {
    let worker_name = &worker.name;
    let worker_instructions = &worker.instructions;

    let mut prompt = format!(
        "You are worker \"{worker_name}\" in a commander-managed Codex workflow.\n\
Role instructions:\n{worker_instructions}\n\n\
Shared task:\n{task}\n"
    );

    let prev = select_previous_outputs(previous_outputs, worker_name, config);
    if prev.is_empty() {
        prompt.push_str("\nNo previous worker outputs are available.\n");
    } else {
        prompt.push_str(&format_previous_outputs_section(
            prev,
            config.max_context_chars,
        ));
    }

    prompt.push_str(
        "\nProvide only your actionable output for the next worker and keep it concrete.\n",
    );

    prompt
}

fn select_previous_outputs<'a>(
    previous_outputs: &'a [(String, String)],
    worker_name: &str,
    config: &CommanderRunConfig,
) -> &'a [(String, String)] {
    // For reviewer/synthesizer, context is the whole point. Prefer full history (then cap by max_context_chars).
    if worker_name.eq_ignore_ascii_case("reviewer")
        || worker_name.eq_ignore_ascii_case("synthesizer")
    {
        return previous_outputs;
    }

    match config.previous_outputs {
        PreviousOutputsSelection::All => previous_outputs,
        PreviousOutputsSelection::None => &[],
        PreviousOutputsSelection::Last(n) => {
            if n == 0 || previous_outputs.is_empty() {
                &[]
            } else if previous_outputs.len() <= n {
                previous_outputs
            } else {
                &previous_outputs[previous_outputs.len() - n..]
            }
        }
    }
}

fn format_previous_outputs_section(
    previous_outputs: &[(String, String)],
    max_context_chars: Option<usize>,
) -> String {
    if previous_outputs.is_empty() {
        return "\nNo previous worker outputs are available.\n".to_string();
    }

    let mut out = String::new();
    out.push_str("\nPrevious worker outputs:\n");

    match max_context_chars {
        None => {
            for (name, output) in previous_outputs {
                out.push_str(&format!("\n[{name}]\n{output}\n"));
            }
        }
        Some(budget) => {
            // Budget applies to the content added from previous outputs (not the header line).
            let mut remaining = budget;

            // Select from the newest backwards, then print in chronological order.
            let mut selected: Vec<(String, String)> = Vec::new();

            for (name, output) in previous_outputs.iter().rev() {
                if remaining == 0 {
                    break;
                }

                let header = format!("\n[{name}]\n");
                let footer = "\n".to_string();

                let header_chars = header.chars().count() + footer.chars().count();
                if header_chars >= remaining {
                    // Can't even fit a header + newline; stop.
                    break;
                }

                let available_for_output = remaining - header_chars;
                let output_snippet = truncate_output_to_budget(output, available_for_output);

                let segment_chars = header.chars().count()
                    + output_snippet.chars().count()
                    + footer.chars().count();

                if segment_chars > remaining {
                    // Shouldn't happen due to truncation, but keep it defensive.
                    break;
                }

                remaining -= segment_chars;
                selected.push((name.clone(), output_snippet));
            }

            selected.reverse();
            for (name, snippet) in selected {
                out.push_str(&format!("\n[{name}]\n{snippet}\n"));
            }

            if previous_outputs.len() > 1 && budget > 0 && remaining == 0 {
                out.push_str("\n[context budget reached]\n");
            }
        }
    }

    out
}

fn truncate_output_to_budget(output: &str, budget_chars: usize) -> String {
    if budget_chars == 0 {
        return String::new();
    }

    let output_chars = output.chars().count();
    if output_chars <= budget_chars {
        return output.to_string();
    }

    // Reserve space for a truncation marker.
    const MARK: &str = "\n[truncated]";
    let mark_len = MARK.chars().count();

    if budget_chars <= mark_len {
        return output.chars().take(budget_chars).collect();
    }

    let take = budget_chars - mark_len;
    let mut s: String = output.chars().take(take).collect();
    s.push_str(MARK);
    s
}

async fn run_commander_shell(
    session: &mut CommanderSession,
    root_config_overrides: &CliConfigOverrides,
    codex_bin: &Path,
) -> anyhow::Result<()> {
    let mut stdout = tokio::io::stdout();
    reset_shell_surface(&mut stdout).await?;

    println!("Commander shell started. Use /help for commands.");
    println!("Type a task directly (no /task needed).");
    print_shell_dashboard(session);

    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();

    loop {
        let prompt = shell_prompt(session);
        let rendered_bottom = render_bottom_console(&mut stdout, session, &prompt)
            .await
            .context("failed to render bottom console")?;
        if !rendered_bottom {
            stdout
                .write_all(prompt.as_bytes())
                .await
                .context("failed to write commander prompt")?;
            stdout
                .flush()
                .await
                .context("failed to flush commander prompt")?;
        }

        let Some(input) = lines
            .next_line()
            .await
            .context("failed to read commander command")?
        else {
            stdout
                .write_all(b"\x1b[0m\x1b[?2004l")
                .await
                .context("failed to reset terminal style")?;
            println!();
            break;
        };
        stdout
            .write_all(b"\x1b[0m")
            .await
            .context("failed to reset terminal style")?;

        let normalized = read_shell_input_with_paste_support(&mut lines, input).await?;
        match parse_shell_command(&normalized) {
            CommanderShellCommand::Task(task) => {
                let task_summary = summarize_task_input(&task);
                if let Some(preview) = summarize_large_task_preview(&task) {
                    println!("Task input: {preview}");
                }
                session.task = Some(task);
                session.worker_outputs.clear();
                refresh_session_worker_states(session);
                println!("Task updated {task_summary}.");
            }
            CommanderShellCommand::AttachImage(path) => {
                let marker = append_image_marker_to_task(session, path.as_deref());
                let updated_task = session.task.as_deref().unwrap_or_default();
                let task_summary = summarize_task_input(updated_task);
                if let Some(preview) = summarize_large_task_preview(updated_task) {
                    println!("Task input: {preview}");
                }
                println!("Attached image marker: {marker}");
                println!("Task updated {task_summary}.");
            }
            CommanderShellCommand::PasteClipboardImage => match paste_image_to_temp_png() {
                Ok((path, info)) => {
                    let path_display = path.display().to_string();
                    let marker = append_image_marker_to_task(session, Some(&path_display));
                    let updated_task = session.task.as_deref().unwrap_or_default();
                    let task_summary = summarize_task_input(updated_task);
                    if let Some(preview) = summarize_large_task_preview(updated_task) {
                        println!("Task input: {preview}");
                    }
                    println!(
                        "Attached clipboard image: {}x{} {} -> {marker}",
                        info.width,
                        info.height,
                        info.encoded_format.label()
                    );
                    println!("Task updated {task_summary}.");
                }
                Err(err) => {
                    eprintln!("Failed to paste clipboard image: {err}");
                }
            },
            CommanderShellCommand::AddWorker(spec) => match parse_worker_definition(&spec) {
                Ok(worker) => {
                    let worker_name = worker.name.clone();
                    upsert_worker(&mut session.workers, worker);
                    session.worker_outputs.clear();
                    refresh_session_worker_states(session);
                    println!("Upserted worker '{worker_name}'.");
                }
                Err(err) => {
                    eprintln!("Invalid worker spec: {err}");
                }
            },
            CommanderShellCommand::RemoveWorker(name) => {
                let previous_len = session.workers.len();
                session
                    .workers
                    .retain(|worker| !worker.name.eq_ignore_ascii_case(&name));
                if session.workers.len() == previous_len {
                    eprintln!("No worker named '{name}' was found.");
                } else {
                    println!("Removed worker '{name}'.");
                    session.worker_outputs.clear();
                    refresh_session_worker_states(session);
                }
            }
            CommanderShellCommand::Run => {
                if let Err(err) = execute_workers(session, root_config_overrides, codex_bin).await {
                    eprintln!("Run failed: {err}");
                }
            }
            CommanderShellCommand::Workers => {
                print_worker_statuses(&session.worker_states);
            }
            CommanderShellCommand::Tree => {
                for line in format_worker_tree_lines(session) {
                    println!("{line}");
                }
            }
            CommanderShellCommand::Results => {
                for line in format_results_lines(session) {
                    println!("{line}");
                }
            }
            CommanderShellCommand::Show => {
                print_shell_dashboard(session);
            }
            CommanderShellCommand::Config => {
                for line in format_config_lines(session) {
                    println!("{line}");
                }
            }
            CommanderShellCommand::Save(path) => match save_session(session, Path::new(&path)) {
                Ok(()) => println!("Session saved to {path}"),
                Err(err) => eprintln!("Save failed: {err}"),
            },
            CommanderShellCommand::Load(path) => match load_session_into(session, Path::new(&path))
            {
                Ok(()) => {
                    session.worker_outputs.clear();
                    refresh_session_worker_states(session);
                    println!("Session loaded from {path}");
                }
                Err(err) => eprintln!("Load failed: {err}"),
            },
            CommanderShellCommand::ClearResults => {
                session.worker_outputs.clear();
                refresh_session_worker_states(session);
                println!("Cleared results and reset worker states.");
            }
            CommanderShellCommand::SetModel(model) => {
                session.run_config.model =
                    if model.eq_ignore_ascii_case("unset") || model.eq_ignore_ascii_case("none") {
                        None
                    } else {
                        Some(model)
                    };
                println!("Model updated.");
            }
            CommanderShellCommand::SetProfile(profile) => {
                session.run_config.config_profile = if profile.eq_ignore_ascii_case("unset")
                    || profile.eq_ignore_ascii_case("none")
                {
                    None
                } else {
                    Some(profile)
                };
                println!("Profile updated.");
            }
            CommanderShellCommand::ToggleFullAuto => {
                session.run_config.full_auto = !session.run_config.full_auto;
                if session.run_config.full_auto {
                    session.run_config.yolo = false;
                }
                println!("full-auto: {}", session.run_config.full_auto);
            }
            CommanderShellCommand::ToggleYolo => {
                session.run_config.yolo = !session.run_config.yolo;
                if session.run_config.yolo {
                    session.run_config.full_auto = false;
                }
                println!("yolo: {}", session.run_config.yolo);
            }
            CommanderShellCommand::SetTimeout(secs) => {
                session.run_config.timeout = secs.map(Duration::from_secs);
                println!(
                    "timeout: {}",
                    session
                        .run_config
                        .timeout
                        .map(|d| format!("{}s", d.as_secs()))
                        .unwrap_or_else(|| "unset".to_string())
                );
            }
            CommanderShellCommand::SetRetries(n) => {
                session.run_config.retries = n;
                println!("retries: {}", session.run_config.retries);
            }
            CommanderShellCommand::ToggleContinueOnFailure => {
                session.run_config.continue_on_failure = !session.run_config.continue_on_failure;
                println!(
                    "continue-on-failure: {}",
                    session.run_config.continue_on_failure
                );
            }
            CommanderShellCommand::ToggleAutoReviewer => {
                session.run_config.auto_reviewer.enabled =
                    !session.run_config.auto_reviewer.enabled;
                refresh_session_worker_states(session);
                println!(
                    "auto-reviewer: {}",
                    session.run_config.auto_reviewer.enabled
                );
            }
            CommanderShellCommand::ToggleFinalize => {
                session.run_config.finalize = !session.run_config.finalize;
                refresh_session_worker_states(session);
                println!("finalize: {}", session.run_config.finalize);
            }
            CommanderShellCommand::SetOutputDir(path) => {
                session.run_config.output_dir = path
                    .filter(|p| !p.eq_ignore_ascii_case("unset") && !p.eq_ignore_ascii_case("none"))
                    .map(PathBuf::from);
                println!(
                    "output-dir: {}",
                    session
                        .run_config
                        .output_dir
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "unset".to_string())
                );
            }
            CommanderShellCommand::ToggleStream => {
                session.run_config.stream = !session.run_config.stream;
                println!("stream: {}", session.run_config.stream);
            }
            CommanderShellCommand::ToggleDryRun => {
                session.run_config.dry_run = !session.run_config.dry_run;
                println!("dry-run: {}", session.run_config.dry_run);
            }
            CommanderShellCommand::Help => {
                println!("Available commands:");
                println!("(plain text without / is treated as /task <text>)");
                println!("/task <text> - set shared task");
                println!("/image [path] - paste clipboard image (no path) or attach path marker");
                println!("/attach-image [path] - append image marker without clipboard paste");
                println!("/image-paste - paste image from clipboard via Codex core pipeline");
                println!("/add-worker <name>=<instructions> - add or update worker");
                println!("/remove-worker <name> - remove worker");
                println!("/run - execute all workers in sequence");
                println!("/workers - show worker statuses");
                println!("/tree - show worker execution tree");
                println!("/results - show completed outputs");
                println!("/show - show current task and workers");
                println!("/config - show current run configuration");
                println!("/save <path> - save task+workers as a session JSON");
                println!("/load <path> - load task+workers from a session JSON");
                println!("/clear-results - clear outputs and reset state");
                println!("/set-model <model|unset> - set/clear model override");
                println!("/set-profile <profile|unset> - set/clear profile override");
                println!("/toggle-full-auto - toggle full-auto (disables yolo)");
                println!("/toggle-yolo - toggle yolo (disables full-auto)");
                println!("/set-timeout <secs|unset> - set/clear worker timeout");
                println!("/set-retries <n> - set retries per worker");
                println!("/toggle-continue-on-failure - toggle continue-on-failure");
                println!("/toggle-auto-reviewer - toggle auto reviewer injection");
                println!("/toggle-finalize - toggle synthesizer injection");
                println!("/set-output-dir <path|unset> - set/clear output dir");
                println!("/toggle-stream - toggle streaming stdout/stderr");
                println!("/toggle-dry-run - toggle dry-run mode");
                println!("? - show this help");
                println!("/help - show this help");
                println!("/quit - exit commander shell");
            }
            CommanderShellCommand::Quit => break,
            CommanderShellCommand::Empty => {}
            CommanderShellCommand::UsageError(message) => {
                eprintln!("{message}");
            }
            CommanderShellCommand::Unknown(command) => {
                eprintln!(
                    "Unknown command: {command}. Use /help. If this is a task, type it without a leading '/'."
                );
            }
        }
    }

    stdout
        .write_all(b"\x1b[0m\x1b[?2004l")
        .await
        .context("failed to restore terminal paste mode")?;
    stdout
        .flush()
        .await
        .context("failed to flush terminal paste mode restore")?;

    Ok(())
}

fn parse_shell_command(input: &str) -> CommanderShellCommand {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return CommanderShellCommand::Empty;
    }
    if trimmed == "?" {
        return CommanderShellCommand::Help;
    }

    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    let args = parts.next().unwrap_or_default().trim();

    match command {
        "/task" => {
            if args.is_empty() {
                CommanderShellCommand::UsageError("Usage: /task <task text>".to_string())
            } else {
                CommanderShellCommand::Task(args.to_string())
            }
        }
        "/image" => {
            if args.is_empty() {
                CommanderShellCommand::PasteClipboardImage
            } else {
                CommanderShellCommand::AttachImage(Some(args.to_string()))
            }
        }
        "/attach-image" => {
            if args.is_empty() {
                CommanderShellCommand::AttachImage(None)
            } else {
                CommanderShellCommand::AttachImage(Some(args.to_string()))
            }
        }
        "/image-paste" | "/paste-image" => {
            if args.is_empty() {
                CommanderShellCommand::PasteClipboardImage
            } else {
                CommanderShellCommand::UsageError("Usage: /image-paste".to_string())
            }
        }
        "/add-worker" => {
            if args.is_empty() {
                CommanderShellCommand::UsageError(
                    "Usage: /add-worker <name>=<instructions>".to_string(),
                )
            } else {
                CommanderShellCommand::AddWorker(args.to_string())
            }
        }
        "/remove-worker" => {
            if args.is_empty() {
                CommanderShellCommand::UsageError("Usage: /remove-worker <name>".to_string())
            } else {
                CommanderShellCommand::RemoveWorker(args.to_string())
            }
        }
        "/run" | "/r" => CommanderShellCommand::Run,
        "/workers" | "/w" => CommanderShellCommand::Workers,
        "/tree" | "/t" => CommanderShellCommand::Tree,
        "/results" | "/res" => CommanderShellCommand::Results,
        "/show" | "/s" => CommanderShellCommand::Show,
        "/config" | "/cfg" => CommanderShellCommand::Config,
        "/save" => {
            if args.is_empty() {
                CommanderShellCommand::UsageError("Usage: /save <path>".to_string())
            } else {
                CommanderShellCommand::Save(args.to_string())
            }
        }
        "/load" => {
            if args.is_empty() {
                CommanderShellCommand::UsageError("Usage: /load <path>".to_string())
            } else {
                CommanderShellCommand::Load(args.to_string())
            }
        }
        "/clear-results" | "/clear" => CommanderShellCommand::ClearResults,
        "/set-model" => {
            if args.is_empty() {
                CommanderShellCommand::UsageError("Usage: /set-model <model|unset>".to_string())
            } else {
                CommanderShellCommand::SetModel(args.to_string())
            }
        }
        "/set-profile" => {
            if args.is_empty() {
                CommanderShellCommand::UsageError("Usage: /set-profile <profile|unset>".to_string())
            } else {
                CommanderShellCommand::SetProfile(args.to_string())
            }
        }
        "/toggle-full-auto" => CommanderShellCommand::ToggleFullAuto,
        "/toggle-yolo" => CommanderShellCommand::ToggleYolo,
        "/set-timeout" => {
            if args.is_empty()
                || args.eq_ignore_ascii_case("unset")
                || args.eq_ignore_ascii_case("none")
            {
                CommanderShellCommand::SetTimeout(None)
            } else {
                match args.parse::<u64>() {
                    Ok(n) => CommanderShellCommand::SetTimeout(Some(n)),
                    Err(_) => CommanderShellCommand::UsageError(
                        "Usage: /set-timeout <secs|unset>".to_string(),
                    ),
                }
            }
        }
        "/set-retries" => {
            if args.is_empty() {
                CommanderShellCommand::UsageError("Usage: /set-retries <n>".to_string())
            } else {
                match args.parse::<usize>() {
                    Ok(n) => CommanderShellCommand::SetRetries(n),
                    Err(_) => {
                        CommanderShellCommand::UsageError("Usage: /set-retries <n>".to_string())
                    }
                }
            }
        }
        "/toggle-continue-on-failure" => CommanderShellCommand::ToggleContinueOnFailure,
        "/toggle-auto-reviewer" => CommanderShellCommand::ToggleAutoReviewer,
        "/toggle-finalize" => CommanderShellCommand::ToggleFinalize,
        "/set-output-dir" => {
            if args.is_empty() {
                CommanderShellCommand::UsageError("Usage: /set-output-dir <path|unset>".to_string())
            } else if args.eq_ignore_ascii_case("unset") || args.eq_ignore_ascii_case("none") {
                CommanderShellCommand::SetOutputDir(None)
            } else {
                CommanderShellCommand::SetOutputDir(Some(args.to_string()))
            }
        }
        "/toggle-stream" => CommanderShellCommand::ToggleStream,
        "/toggle-dry-run" => CommanderShellCommand::ToggleDryRun,
        "/help" | "/h" => CommanderShellCommand::Help,
        "/quit" | "/exit" | "/q" => CommanderShellCommand::Quit,
        _ => {
            if command.starts_with('/') {
                CommanderShellCommand::Unknown(trimmed.to_string())
            } else {
                CommanderShellCommand::Task(trimmed.to_string())
            }
        }
    }
}

fn format_worker_status_lines(worker_states: &[WorkerRuntimeState]) -> Vec<String> {
    let mut lines = vec!["Worker status:".to_string()];
    for state in worker_states {
        let worker_name = &state.name;
        let status_label = state.status.label();

        let mut suffixes: Vec<String> = Vec::new();
        if state.output_chars > 0 {
            suffixes.push(format!("{} chars", state.output_chars));
        }
        if state.attempts > 0 && state.status.is_done() {
            suffixes.push(format!("{} attempt(s)", state.attempts));
        }
        if let Some(ms) = state.duration_ms {
            if state.status.is_done() || state.status == WorkerStatus::Running {
                suffixes.push(format!("{ms}ms"));
            }
        }

        if suffixes.is_empty() {
            lines.push(format!("- {worker_name}: {status_label}"));
        } else {
            lines.push(format!(
                "- {worker_name}: {status_label} ({})",
                suffixes.join(", ")
            ));
        }
    }

    lines
}

fn print_worker_statuses(worker_states: &[WorkerRuntimeState]) {
    for line in format_worker_status_lines(worker_states) {
        println!("{line}");
    }
}

fn format_progress_title(worker_states: &[WorkerRuntimeState]) -> String {
    if worker_states.is_empty() {
        return "0/0".to_string();
    }

    let total = worker_states.len();
    if let Some((index, state)) = worker_states
        .iter()
        .enumerate()
        .find(|(_, state)| state.status == WorkerStatus::Running)
    {
        let worker_name = &state.name;
        return format!("{}/{} running ({worker_name})", index + 1, total);
    }

    let completed = worker_states.iter().filter(|s| s.status.is_done()).count();

    if completed == total {
        let failed = worker_states
            .iter()
            .filter(|state| matches!(state.status, WorkerStatus::Failed | WorkerStatus::TimedOut))
            .count();
        if failed == 0 {
            format!("{completed}/{total} done")
        } else {
            format!("{completed}/{total} done ({failed} failed)")
        }
    } else {
        format!("{completed}/{total}")
    }
}

fn shell_prompt(session: &CommanderSession) -> String {
    let progress_title = format_progress_title(&session.worker_states);
    format!("› [{progress_title}] ")
}

const DASHBOARD_BOX_WIDTH_DEFAULT: usize = 58;
const DASHBOARD_BOX_WIDTH_MIN: usize = 48;
const DASHBOARD_BOX_WIDTH_MAX: usize = 76;
const STYLED_FOOTER_MIN_ROWS: usize = 18;
const PASTED_PREVIEW_MIN_CHARS: usize = 50;
const PASTE_BURST_IDLE_MS: u64 = 120;
const BRACKETED_PASTE_START: &str = "\u{1b}[200~";
const BRACKETED_PASTE_END: &str = "\u{1b}[201~";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DashboardLayout {
    Full,
    Compact,
    Minimal,
}

fn truncate_for_box(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let prefix = text.chars().take(max_chars - 3).collect::<String>();
    format!("{prefix}...")
}

fn truncate_middle_for_box(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let left = (max_chars - 3) / 2;
    let right = max_chars.saturating_sub(3 + left);
    let prefix = text.chars().take(left).collect::<String>();
    let suffix = text
        .chars()
        .rev()
        .take(right)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("{prefix}...{suffix}")
}

fn dashboard_box_width(cols: Option<usize>) -> usize {
    let Some(cols) = cols else {
        return DASHBOARD_BOX_WIDTH_DEFAULT;
    };

    let usable = cols.saturating_sub(2).max(4);
    if usable < DASHBOARD_BOX_WIDTH_MIN {
        usable
    } else {
        usable.clamp(DASHBOARD_BOX_WIDTH_MIN, DASHBOARD_BOX_WIDTH_MAX)
    }
}

fn format_box_border_with_width(top: bool, box_width: usize) -> String {
    let inner = "─".repeat(box_width.saturating_sub(2));
    if top {
        format!("┌{inner}┐")
    } else {
        format!("└{inner}┘")
    }
}

#[cfg(test)]
fn format_box_border(top: bool) -> String {
    format_box_border_with_width(top, DASHBOARD_BOX_WIDTH_DEFAULT)
}

fn format_box_line_with_width(content: &str, box_width: usize) -> String {
    let inner = box_width.saturating_sub(2);
    let clipped = truncate_for_box(content, inner);
    let padding = " ".repeat(inner.saturating_sub(clipped.chars().count()));
    format!("│{clipped}{padding}│")
}

fn pick_dashboard_layout(rows: Option<usize>) -> DashboardLayout {
    match rows {
        Some(rows) if rows >= 20 => DashboardLayout::Full,
        Some(rows) if rows >= 14 => DashboardLayout::Compact,
        _ => DashboardLayout::Minimal,
    }
}

fn read_tput_number(capability: &str) -> Option<usize> {
    std::process::Command::new("tput")
        .arg(capability)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<usize>()
                .ok()
        })
}

fn terminal_dimensions() -> Option<(usize, usize)> {
    let rows = read_tput_number("lines")?;
    let cols = read_tput_number("cols")?;
    if rows == 0 || cols == 0 {
        None
    } else {
        Some((rows, cols))
    }
}

fn fit_text_to_width(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let clipped = text.chars().take(width).collect::<String>();
    let clipped_width = clipped.chars().count();
    if clipped_width >= width {
        clipped
    } else {
        format!("{clipped}{}", " ".repeat(width - clipped_width))
    }
}

fn format_footer_status_line(progress_title: &str, width: usize) -> String {
    let left = "? for shortcuts";
    let right = format!("{progress_title} context left");
    let left_width = left.chars().count();
    let right_width = right.chars().count();
    if left_width + 1 + right_width <= width {
        let spaces = width - left_width - right_width;
        format!("{left}{}{right}", " ".repeat(spaces))
    } else {
        fit_text_to_width(left, width)
    }
}

fn footer_input_rows(rows: usize) -> usize {
    if rows >= 36 { 5 } else { 4 }
}

async fn reset_shell_surface(stdout: &mut tokio::io::Stdout) -> anyhow::Result<()> {
    stdout
        .write_all(b"\x1b[0m\x1b[?2004l\x1b[2J\x1b[H")
        .await
        .context("failed to clear commander shell surface")?;
    stdout
        .flush()
        .await
        .context("failed to flush commander shell surface")
}

fn normalize_shell_input(input: &str) -> String {
    input
        .replace(BRACKETED_PASTE_START, "")
        .replace(BRACKETED_PASTE_END, "")
}

fn split_bracketed_paste_markers(input: &str) -> (bool, bool, String) {
    let has_start = input.contains(BRACKETED_PASTE_START);
    let has_end = input.contains(BRACKETED_PASTE_END);
    (has_start, has_end, normalize_shell_input(input))
}

async fn read_shell_input_with_paste_support(
    lines: &mut tokio::io::Lines<BufReader<tokio::io::Stdin>>,
    first_line: String,
) -> anyhow::Result<String> {
    let (has_start, has_end, normalized) = split_bracketed_paste_markers(&first_line);
    if !has_start {
        return Ok(normalized);
    }
    if has_end {
        return Ok(normalized);
    }

    if normalized.trim_start().starts_with('/') {
        return Ok(normalized);
    }

        let mut collected = normalized;
        let is_burst_collection = !has_start;

    loop {
            let next_line = if is_burst_collection {
            match timeout(
                std::time::Duration::from_millis(PASTE_BURST_IDLE_MS),
                lines.next_line(),
            )
            .await
            {
                Ok(next) => next,
                Err(_) => break,
            }
        } else {
            lines.next_line().await
        };

        let Some(next) = next_line.context("failed to read pasted commander command")? else {
            break;
        };
        let (_, end_found, cleaned) = split_bracketed_paste_markers(&next);
        if !collected.is_empty() {
            collected.push('\n');
        }
        collected.push_str(&cleaned);
        if end_found {
            break;
        }
    }

    Ok(collected)
}

fn summarize_task_input(task: &str) -> String {
    let chars = task.chars().count();
    let lower = task.to_ascii_lowercase();
    let image_like = lower.contains("[image #")
        || lower.starts_with("data:image/")
        || lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".webp")
        || lower.ends_with(".gif");
    if image_like {
        format!("[image] [#char {chars}]")
    } else {
        format!("[#char {chars}]")
    }
}

fn append_image_marker_to_task(session: &mut CommanderSession, image_path: Option<&str>) -> String {
    let image_count = session
        .task
        .as_deref()
        .map(|task| task.matches("[Image #").count())
        .unwrap_or(0)
        + 1;
    let marker = match image_path {
        Some(path) => format!("[Image #{image_count}] {path}"),
        None => format!("[Image #{image_count}]"),
    };
    let next_task = match session.task.take() {
        None => marker.clone(),
        Some(existing) if existing.trim().is_empty() => marker.clone(),
        Some(existing) if existing.ends_with('\n') => format!("{existing}{marker}"),
        Some(existing) => format!("{existing}\n{marker}"),
    };
    session.task = Some(next_task);
    session.worker_outputs.clear();
    refresh_session_worker_states(session);
    marker
}

fn summarize_large_task_preview(task: &str) -> Option<String> {
    let chars = task.chars().count();
    if chars > PASTED_PREVIEW_MIN_CHARS {
        Some(format!("[Pasted Content {chars} chars]"))
    } else {
        None
    }
}

fn footer_tip_line() -> &'static str {
    "Tip: type task text directly; use /run when ready. /help for commands."
}

fn format_directory_for_display(path: &Path) -> String {
    let rendered = path.display().to_string();
    let Ok(home) = std::env::var("HOME") else {
        return rendered;
    };
    if rendered == home {
        "~".to_string()
    } else if let Some(suffix) = rendered.strip_prefix(&format!("{home}/")) {
        format!("~/{suffix}")
    } else {
        rendered
    }
}

async fn render_bottom_console(
    stdout: &mut tokio::io::Stdout,
    session: &CommanderSession,
    prompt: &str,
) -> anyhow::Result<bool> {
    let Some((rows, cols)) = terminal_dimensions() else {
        return Ok(false);
    };
    if rows < STYLED_FOOTER_MIN_ROWS || cols == 0 {
        return Ok(false);
    }

    let input_rows = footer_input_rows(rows);
    let reserved_rows = input_rows + 3; // tip + spacer + input + status
    if rows < reserved_rows {
        return Ok(false);
    }

    let tip_row = rows - reserved_rows + 1;
    let spacer_row = tip_row + 1;
    let input_top_row = spacer_row + 1;
    let input_bottom_row = input_top_row + input_rows - 1;
    let prompt_row = input_top_row + (input_rows.saturating_sub(1) / 2);
    let status_row = rows.max(1);
    let progress_title = format_progress_title(&session.worker_states);
    let tip_line = fit_text_to_width(footer_tip_line(), cols);
    let spacer_line = fit_text_to_width("", cols);
    let input_fill_line = fit_text_to_width("", cols);
    let prompt_prefix = format!(" {prompt}");
    let prompt_line = fit_text_to_width(&prompt_prefix, cols);
    let status_line = fit_text_to_width(&format_footer_status_line(&progress_title, cols), cols);
    let prompt_col = prompt_prefix.chars().count().min(cols.saturating_sub(1)) + 1;

    let mut frame = String::new();
    frame.push_str(&format!(
        "\x1b[{tip_row};1H\x1b[2K\x1b[38;5;255m{tip_line}\x1b[0m"
    ));
    frame.push_str(&format!("\x1b[{spacer_row};1H\x1b[2K{spacer_line}"));

    for row in input_top_row..=input_bottom_row {
        let line = if row == prompt_row {
            &prompt_line
        } else {
            &input_fill_line
        };
        frame.push_str(&format!(
            "\x1b[{row};1H\x1b[2K\x1b[48;5;236m\x1b[38;5;255m{line}\x1b[0m"
        ));
    }

    frame.push_str(&format!(
        "\x1b[{status_row};1H\x1b[2K\x1b[48;5;234m\x1b[38;5;245m{status_line}\x1b[0m\
\x1b[48;5;236m\x1b[38;5;255m\x1b[{prompt_row};{prompt_col}H"
    ));
    stdout
        .write_all(frame.as_bytes())
        .await
        .context("failed to write bottom console frame")?;
    stdout
        .flush()
        .await
        .context("failed to flush bottom console frame")?;

    Ok(true)
}

#[cfg(test)]
fn format_shell_dashboard_lines(session: &CommanderSession) -> Vec<String> {
    format_shell_dashboard_lines_with_layout(
        session,
        DashboardLayout::Full,
        DASHBOARD_BOX_WIDTH_DEFAULT,
    )
}

fn format_shell_dashboard_lines_with_layout(
    session: &CommanderSession,
    layout: DashboardLayout,
    box_width: usize,
) -> Vec<String> {
    let version = env!("CARGO_PKG_VERSION");
    let progress_title = format_progress_title(&session.worker_states);
    let mut directory = std::env::current_dir()
        .ok()
        .map(|p| format_directory_for_display(&p))
        .unwrap_or_else(|| "<unknown>".to_string());
    let inner = box_width.saturating_sub(2);
    let directory_budget = inner.saturating_sub("directory: ".chars().count());
    directory = truncate_middle_for_box(&directory, directory_budget);
    let model = session
        .run_config
        .model
        .as_deref()
        .unwrap_or("gpt-5.3-codex xhigh");
    let task_title = session
        .task
        .as_deref()
        .unwrap_or("<unset> (use /task <text>)");
    let configured_workers = session.workers.len();
    let execution_workers = execution_workers(session).len();

    let reviewer_state = if session
        .workers
        .iter()
        .any(|worker| worker.name.eq_ignore_ascii_case("reviewer"))
    {
        "configured"
    } else if session.run_config.auto_reviewer.enabled && !session.workers.is_empty() {
        "auto on /run"
    } else if session.run_config.auto_reviewer.enabled {
        "auto (needs workers)"
    } else {
        "off"
    };

    match layout {
        DashboardLayout::Full => vec![
            format_box_border_with_width(true, box_width),
            format_box_line_with_width(
                &format!(">_ OpenAI Codex Commander (v{version})"),
                box_width,
            ),
            format_box_line_with_width("", box_width),
            format_box_line_with_width(&format!("model:     {model}"), box_width),
            format_box_line_with_width(&format!("directory: {directory}"), box_width),
            format_box_line_with_width("", box_width),
            format_box_line_with_width(&format!("worker:   {progress_title}"), box_width),
            format_box_line_with_width(&format!("task:     {task_title}"), box_width),
            format_box_line_with_width(
                &format!(
                    "workers:  {configured_workers} configured ({execution_workers} run order)"
                ),
                box_width,
            ),
            format_box_line_with_width(&format!("reviewer: {reviewer_state}"), box_width),
            format_box_border_with_width(false, box_width),
        ],
        DashboardLayout::Compact => vec![
            format_box_border_with_width(true, box_width),
            format_box_line_with_width(
                &format!(">_ OpenAI Codex Commander (v{version})"),
                box_width,
            ),
            format_box_line_with_width("", box_width),
            format_box_line_with_width(&format!("model:     {model}"), box_width),
            format_box_line_with_width(&format!("directory: {directory}"), box_width),
            format_box_line_with_width(&format!("worker:   {progress_title}"), box_width),
            format_box_line_with_width(&format!("reviewer: {reviewer_state}"), box_width),
            format_box_border_with_width(false, box_width),
        ],
        DashboardLayout::Minimal => vec![
            format_box_border_with_width(true, box_width),
            format_box_line_with_width(
                &format!(">_ OpenAI Codex Commander (v{version})"),
                box_width,
            ),
            format_box_line_with_width(&format!("model:     {model}"), box_width),
            format_box_line_with_width(&format!("directory: {directory}"), box_width),
            format_box_border_with_width(false, box_width),
        ],
    }
}

fn print_shell_dashboard(session: &CommanderSession) {
    let dimensions = terminal_dimensions();
    let layout = pick_dashboard_layout(dimensions.map(|(rows, _)| rows));
    let box_width = dashboard_box_width(dimensions.map(|(_, cols)| cols));
    let lines = format_shell_dashboard_lines_with_layout(session, layout, box_width);
    println!();
    for line in lines {
        println!("{line}");
    }
    println!();
    println!();
}

fn format_config_lines(session: &CommanderSession) -> Vec<String> {
    let cfg = &session.run_config;
    let model = cfg.model.as_deref().unwrap_or("<inherit>");
    let profile = cfg.config_profile.as_deref().unwrap_or("<inherit>");
    let timeout = cfg
        .timeout
        .map(|d| format!("{}s", d.as_secs()))
        .unwrap_or_else(|| "<unset>".to_string());

    let prev = match cfg.previous_outputs {
        PreviousOutputsSelection::All => "all".to_string(),
        PreviousOutputsSelection::None => "none".to_string(),
        PreviousOutputsSelection::Last(n) => format!("last {n}"),
    };

    let max_ctx = cfg
        .max_context_chars
        .map(|n| n.to_string())
        .unwrap_or_else(|| "<unset>".to_string());

    let outdir = cfg
        .output_dir
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<unset>".to_string());

    vec![
        "Run config:".to_string(),
        format!("- model: {model}"),
        format!("- profile: {profile}"),
        format!("- full-auto: {}", cfg.full_auto),
        format!("- yolo: {}", cfg.yolo),
        format!("- stream: {}", cfg.stream),
        format!("- dry-run: {}", cfg.dry_run),
        format!("- timeout: {timeout}"),
        format!("- retries: {}", cfg.retries),
        format!("- continue-on-failure: {}", cfg.continue_on_failure),
        format!("- previous-outputs: {prev}"),
        format!("- max-context-chars: {max_ctx}"),
        format!("- auto-reviewer: {}", cfg.auto_reviewer.enabled),
        format!("- finalize: {}", cfg.finalize),
        format!("- output-dir: {outdir}"),
        format!("- save-prompts: {}", cfg.save_prompts),
        format!("- write-report: {}", cfg.write_report),
        format!("- env: {} entries", cfg.env.len()),
    ]
}

fn format_worker_tree_lines(session: &CommanderSession) -> Vec<String> {
    let execution_workers = execution_workers(session);
    if execution_workers.is_empty() {
        return vec!["Worker tree: <none> (use /add-worker name=instructions)".to_string()];
    }

    let reviewer_auto = session.run_config.auto_reviewer.enabled
        && !session
            .workers
            .iter()
            .any(|worker| worker.name.eq_ignore_ascii_case("reviewer"));

    let synth_auto = session.run_config.finalize
        && !session
            .workers
            .iter()
            .any(|worker| worker.name.eq_ignore_ascii_case("synthesizer"));

    let mut lines = vec![
        "Worker tree (execution order):".to_string(),
        "root".to_string(),
    ];

    for (index, worker) in execution_workers.iter().enumerate() {
        let prefix = if index + 1 == execution_workers.len() {
            "└─"
        } else {
            "├─"
        };

        let state = session
            .worker_states
            .iter()
            .find(|state| state.name.eq_ignore_ascii_case(&worker.name));

        let status_label = state
            .map(|s| s.status.label().to_string())
            .unwrap_or_else(|| "pending".to_string());

        let mut line = format!("{prefix} {}. {} [{status_label}]", index + 1, worker.name);

        if reviewer_auto && worker.name.eq_ignore_ascii_case("reviewer") {
            line.push_str(" (auto)");
        }
        if synth_auto && worker.name.eq_ignore_ascii_case("synthesizer") {
            line.push_str(" (auto)");
        }

        if let Some(state) = state {
            if state.output_chars > 0 {
                line.push_str(&format!(" ({} chars)", state.output_chars));
            }
        }

        lines.push(line);
    }

    lines
}

fn format_results_lines(session: &CommanderSession) -> Vec<String> {
    if session.worker_outputs.is_empty() {
        return vec!["No completed worker outputs yet. Use /run first.".to_string()];
    }

    let mut lines = vec!["Completed worker outputs:".to_string()];
    for (worker_name, output) in &session.worker_outputs {
        lines.push(format!("\n[{worker_name}]"));
        lines.push(output.to_string());
    }

    lines
}

fn save_session(session: &CommanderSession, path: &Path) -> anyhow::Result<()> {
    let payload = json!({
        "task": session.task,
        "workers": session.workers.iter().map(|w| json!({
            "name": w.name,
            "instructions": w.instructions,
        })).collect::<Vec<_>>(),
    });

    let bytes = serde_json::to_vec_pretty(&payload).context("failed to serialize session")?;
    std::fs::write(path, bytes)
        .with_context(|| format!("failed to write session {}", path.display()))?;
    Ok(())
}

fn load_session_into(session: &mut CommanderSession, path: &Path) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read session {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).context("failed to parse session JSON")?;

    session.task = value
        .get("task")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    session.workers.clear();
    if let Some(workers) = value.get("workers").and_then(|v| v.as_array()) {
        for w in workers {
            let name = w.get("name").and_then(|v| v.as_str()).unwrap_or("").trim();
            let instructions = w
                .get("instructions")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();

            if name.is_empty() || instructions.is_empty() {
                continue;
            }

            upsert_worker(
                &mut session.workers,
                WorkerDefinition {
                    name: name.to_string(),
                    instructions: instructions.to_string(),
                },
            );
        }
    }

    session.worker_outputs.clear();
    refresh_session_worker_states(session);
    Ok(())
}

fn write_reports(output_dir: &Path, session: &CommanderSession) -> anyhow::Result<()> {
    let mut md = String::new();
    md.push_str("# Codex Commander Report\n\n");

    if let Some(task) = session.task.as_deref() {
        md.push_str("## Task\n\n");
        md.push_str(task);
        md.push_str("\n\n");
    }

    md.push_str("## Worker Outputs\n\n");
    for (name, output) in &session.worker_outputs {
        md.push_str(&format!("### {name}\n\n"));
        md.push_str("```text\n");
        md.push_str(output);
        if !output.ends_with('\n') {
            md.push('\n');
        }
        md.push_str("```\n\n");
    }

    let md_path = output_dir.join("report.md");
    std::fs::write(&md_path, md)
        .with_context(|| format!("failed to write {}", md_path.display()))?;

    let json_report = json!({
        "task": session.task,
        "outputs": session.worker_outputs.iter().map(|(name, output)| json!({
            "worker": name,
            "output": output,
        })).collect::<Vec<_>>(),
        "states": session.worker_states.iter().map(|s| json!({
            "worker": s.name,
            "status": s.status.label(),
            "attempts": s.attempts,
            "duration_ms": s.duration_ms,
            "output_chars": s.output_chars,
        })).collect::<Vec<_>>(),
    });

    let json_path = output_dir.join("report.json");
    let bytes =
        serde_json::to_vec_pretty(&json_report).context("failed to serialize report.json")?;
    std::fs::write(&json_path, bytes)
        .with_context(|| format!("failed to write {}", json_path.display()))?;

    Ok(())
}

fn write_worker_snapshot_md(
    output_dir: &Path,
    index_1: usize,
    worker_name: &str,
    output: &str,
) -> anyhow::Result<()> {
    let base = worker_artifact_base_name(index_1, worker_name);
    let path = output_dir.join(format!("{base}_snapshot.md"));
    let mut md = String::new();
    md.push_str(&format!("# {worker_name} snapshot\n\n"));
    md.push_str("```text\n");
    md.push_str(output);
    if !output.ends_with('\n') {
        md.push('\n');
    }
    md.push_str("```\n");
    std::fs::write(&path, md).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use tempfile::NamedTempFile;

    #[test]
    fn parse_worker_definition_accepts_valid_spec() {
        let parsed = parse_worker_definition("planner=Break the problem down").expect("parse");
        assert_eq!(
            parsed,
            WorkerDefinition {
                name: "planner".to_string(),
                instructions: "Break the problem down".to_string(),
            }
        );
    }

    #[test]
    fn parse_worker_definition_rejects_missing_separator() {
        let err = parse_worker_definition("planner").expect_err("should fail");
        assert_eq!(err.to_string(), "expected NAME=INSTRUCTIONS");
    }

    #[test]
    fn parse_worker_definition_rejects_empty_name() {
        let err = parse_worker_definition("=do work").expect_err("should fail");
        assert_eq!(err.to_string(), "worker name cannot be empty");
    }

    #[test]
    fn parse_worker_definition_rejects_empty_instructions() {
        let err = parse_worker_definition("planner=").expect_err("should fail");
        assert_eq!(err.to_string(), "worker instructions cannot be empty");
    }

    #[test]
    fn parse_env_kv_accepts_empty_value() {
        let (k, v) = parse_env_kv("FOO=").expect("parse");
        assert_eq!(k, "FOO");
        assert_eq!(v, "");
    }

    #[test]
    fn parse_env_kv_rejects_missing_equal() {
        let err = parse_env_kv("FOO").expect_err("should fail");
        assert_eq!(err.to_string(), "expected KEY=VALUE");
    }

    #[test]
    fn parse_env_kv_rejects_empty_key() {
        let err = parse_env_kv("=x").expect_err("should fail");
        assert_eq!(err.to_string(), "env var key cannot be empty");
    }

    #[test]
    fn build_worker_prompt_includes_previous_outputs() {
        let worker = WorkerDefinition {
            name: "coder".to_string(),
            instructions: "Implement changes".to_string(),
        };
        let cfg = CommanderRunConfig::default();
        let prompt = build_worker_prompt(
            "Add a commander command",
            &worker,
            &[("planner".to_string(), "Use clap subcommand".to_string())],
            &cfg,
        );

        assert!(prompt.contains("You are worker \"coder\""));
        assert!(prompt.contains("[planner]"));
        assert!(prompt.contains("Use clap subcommand"));
    }

    #[test]
    fn select_previous_outputs_respects_last_n() {
        let worker = "coder";
        let cfg = CommanderRunConfig {
            previous_outputs: PreviousOutputsSelection::Last(1),
            ..CommanderRunConfig::default()
        };
        let prev = vec![
            ("a".to_string(), "out1".to_string()),
            ("b".to_string(), "out2".to_string()),
        ];
        let selected = select_previous_outputs(&prev, worker, &cfg);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].0, "b");
    }

    #[test]
    fn select_previous_outputs_none_returns_empty() {
        let worker = "coder";
        let cfg = CommanderRunConfig {
            previous_outputs: PreviousOutputsSelection::None,
            ..CommanderRunConfig::default()
        };
        let prev = vec![("a".to_string(), "out1".to_string())];
        let selected = select_previous_outputs(&prev, worker, &cfg);
        assert!(selected.is_empty());
    }

    #[test]
    fn reviewer_always_sees_all_previous_outputs() {
        let cfg = CommanderRunConfig {
            previous_outputs: PreviousOutputsSelection::None,
            ..CommanderRunConfig::default()
        };
        let prev = vec![
            ("a".to_string(), "out1".to_string()),
            ("b".to_string(), "out2".to_string()),
        ];
        let selected = select_previous_outputs(&prev, "reviewer", &cfg);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn format_previous_outputs_section_truncates_by_budget() {
        let prev = vec![
            ("a".to_string(), "0123456789".repeat(10)), // 100 chars
            ("b".to_string(), "X".repeat(100)),
        ];
        let section = format_previous_outputs_section(&prev, Some(80));
        // Should contain b (newest) at least.
        assert!(section.contains("[b]"));
        // Should not explode.
        assert!(section.len() > 0);
    }

    #[test]
    fn parse_shell_command_supports_workers_alias() {
        assert_eq!(
            parse_shell_command("/workers"),
            CommanderShellCommand::Workers
        );
        assert_eq!(parse_shell_command("/w"), CommanderShellCommand::Workers);
        assert_eq!(parse_shell_command("/tree"), CommanderShellCommand::Tree);
        assert_eq!(parse_shell_command("/t"), CommanderShellCommand::Tree);
        assert_eq!(
            parse_shell_command("/results"),
            CommanderShellCommand::Results
        );
        assert_eq!(parse_shell_command("/res"), CommanderShellCommand::Results);
    }

    #[test]
    fn parse_shell_command_supports_help_and_quit() {
        assert_eq!(parse_shell_command("/help"), CommanderShellCommand::Help);
        assert_eq!(parse_shell_command("/h"), CommanderShellCommand::Help);
        assert_eq!(parse_shell_command("?"), CommanderShellCommand::Help);
        assert_eq!(parse_shell_command("/quit"), CommanderShellCommand::Quit);
        assert_eq!(parse_shell_command("/exit"), CommanderShellCommand::Quit);
        assert_eq!(parse_shell_command("/q"), CommanderShellCommand::Quit);
    }

    #[test]
    fn parse_shell_command_parses_task_command() {
        assert_eq!(
            parse_shell_command("/task build auth flow"),
            CommanderShellCommand::Task("build auth flow".to_string())
        );
    }

    #[test]
    fn parse_shell_command_parses_add_worker_command() {
        assert_eq!(
            parse_shell_command("/add-worker planner=make a plan"),
            CommanderShellCommand::AddWorker("planner=make a plan".to_string())
        );
    }

    #[test]
    fn parse_shell_command_reports_usage_errors() {
        assert_eq!(
            parse_shell_command("/task"),
            CommanderShellCommand::UsageError("Usage: /task <task text>".to_string())
        );
        assert_eq!(
            parse_shell_command("/add-worker"),
            CommanderShellCommand::UsageError(
                "Usage: /add-worker <name>=<instructions>".to_string()
            )
        );
        assert_eq!(
            parse_shell_command("/remove-worker"),
            CommanderShellCommand::UsageError("Usage: /remove-worker <name>".to_string())
        );
        assert_eq!(
            parse_shell_command("/save"),
            CommanderShellCommand::UsageError("Usage: /save <path>".to_string())
        );
        assert_eq!(
            parse_shell_command("/load"),
            CommanderShellCommand::UsageError("Usage: /load <path>".to_string())
        );
        assert_eq!(
            parse_shell_command("/image-paste now"),
            CommanderShellCommand::UsageError("Usage: /image-paste".to_string())
        );
    }

    #[test]
    fn parse_shell_command_unknown_is_preserved() {
        assert_eq!(
            parse_shell_command("/something-else"),
            CommanderShellCommand::Unknown("/something-else".to_string())
        );
    }

    #[test]
    fn parse_shell_command_plain_text_is_task() {
        assert_eq!(
            parse_shell_command("investigate live updater"),
            CommanderShellCommand::Task("investigate live updater".to_string())
        );
    }

    #[test]
    fn parse_shell_command_parses_image_command() {
        assert_eq!(
            parse_shell_command("/image ./screen.png"),
            CommanderShellCommand::AttachImage(Some("./screen.png".to_string()))
        );
        assert_eq!(
            parse_shell_command("/attach-image /tmp/a.jpg"),
            CommanderShellCommand::AttachImage(Some("/tmp/a.jpg".to_string()))
        );
        assert_eq!(
            parse_shell_command("/attach-image"),
            CommanderShellCommand::AttachImage(None)
        );
    }

    #[test]
    fn parse_shell_command_parses_clipboard_image_commands() {
        assert_eq!(
            parse_shell_command("/image"),
            CommanderShellCommand::PasteClipboardImage
        );
        assert_eq!(
            parse_shell_command("/image-paste"),
            CommanderShellCommand::PasteClipboardImage
        );
        assert_eq!(
            parse_shell_command("/paste-image"),
            CommanderShellCommand::PasteClipboardImage
        );
    }

    #[test]
    fn normalize_shell_input_removes_bracketed_paste_markers() {
        let input = "\u{1b}[200~fix this ts error\u{1b}[201~";
        assert_eq!(normalize_shell_input(input), "fix this ts error");
    }

    #[test]
    fn summarize_large_task_preview_only_for_content_over_50_chars() {
        assert_eq!(summarize_large_task_preview("small task"), None);
        let exact = "x".repeat(PASTED_PREVIEW_MIN_CHARS);
        assert_eq!(summarize_large_task_preview(&exact), None);
        let long = "x".repeat(PASTED_PREVIEW_MIN_CHARS + 1);
        assert_eq!(
            summarize_large_task_preview(&long),
            Some("[Pasted Content 51 chars]".to_string())
        );
    }

    #[test]
    fn format_worker_status_lines_shows_output_length_for_completed_workers() {
        let worker_states = vec![
            WorkerRuntimeState {
                name: "planner".to_string(),
                status: WorkerStatus::Succeeded,
                output_chars: 42,
                attempts: 1,
                duration_ms: Some(12),
            },
            WorkerRuntimeState {
                name: "coder".to_string(),
                status: WorkerStatus::Running,
                output_chars: 0,
                attempts: 0,
                duration_ms: None,
            },
        ];

        assert_eq!(
            format_worker_status_lines(&worker_states),
            vec![
                "Worker status:".to_string(),
                "- planner: succeeded (42 chars, 1 attempt(s), 12ms)".to_string(),
                "- coder: running".to_string(),
            ]
        );
    }

    #[test]
    fn format_shell_dashboard_lines_shows_unset_task_and_workers_hint() {
        let session = CommanderSession::default();
        let lines = format_shell_dashboard_lines(&session);

        assert_eq!(lines.first(), Some(&format_box_border(true)));
        assert_eq!(lines.last(), Some(&format_box_border(false)));
        assert!(
            lines
                .iter()
                .any(|line| line.contains(">_ OpenAI Codex Commander (v"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("model:     gpt-5.3-codex xhigh"))
        );
        assert!(lines.iter().any(|line| line.contains("directory: ")));
        assert!(lines.iter().any(|line| line.contains("worker:   0/0")));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("task:     <unset> (use /task <text>)"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("workers:  0 configured (0 run order)"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("reviewer: auto (needs workers)"))
        );
    }

    #[test]
    fn format_worker_tree_lines_includes_auto_reviewer() {
        let mut session = CommanderSession {
            workers: vec![
                WorkerDefinition {
                    name: "planner".to_string(),
                    instructions: "Plan".to_string(),
                },
                WorkerDefinition {
                    name: "coder".to_string(),
                    instructions: "Code".to_string(),
                },
            ],
            worker_states: vec![
                WorkerRuntimeState {
                    name: "planner".to_string(),
                    status: WorkerStatus::Succeeded,
                    output_chars: 12,
                    attempts: 1,
                    duration_ms: Some(1),
                },
                WorkerRuntimeState {
                    name: "coder".to_string(),
                    status: WorkerStatus::Pending,
                    output_chars: 0,
                    attempts: 0,
                    duration_ms: None,
                },
                WorkerRuntimeState {
                    name: "reviewer".to_string(),
                    status: WorkerStatus::Pending,
                    output_chars: 0,
                    attempts: 0,
                    duration_ms: None,
                },
            ],
            ..Default::default()
        };

        refresh_session_worker_states(&mut session);

        // After refresh, worker_states are regenerated; set statuses we want for the test.
        session.worker_states = vec![
            WorkerRuntimeState {
                name: "planner".to_string(),
                status: WorkerStatus::Succeeded,
                output_chars: 12,
                attempts: 1,
                duration_ms: Some(1),
            },
            WorkerRuntimeState {
                name: "coder".to_string(),
                status: WorkerStatus::Pending,
                output_chars: 0,
                attempts: 0,
                duration_ms: None,
            },
            WorkerRuntimeState {
                name: "reviewer".to_string(),
                status: WorkerStatus::Pending,
                output_chars: 0,
                attempts: 0,
                duration_ms: None,
            },
        ];

        assert_eq!(
            format_worker_tree_lines(&session),
            vec![
                "Worker tree (execution order):".to_string(),
                "root".to_string(),
                "├─ 1. planner [succeeded] (12 chars)".to_string(),
                "├─ 2. coder [pending]".to_string(),
                "└─ 3. reviewer [pending] (auto)".to_string(),
            ]
        );
    }

    #[test]
    fn shell_prompt_shows_running_worker_number() {
        let session = CommanderSession {
            worker_states: vec![
                WorkerRuntimeState {
                    name: "planner".to_string(),
                    status: WorkerStatus::Succeeded,
                    output_chars: 12,
                    attempts: 1,
                    duration_ms: Some(1),
                },
                WorkerRuntimeState {
                    name: "coder".to_string(),
                    status: WorkerStatus::Running,
                    output_chars: 0,
                    attempts: 0,
                    duration_ms: None,
                },
                WorkerRuntimeState {
                    name: "reviewer".to_string(),
                    status: WorkerStatus::Pending,
                    output_chars: 0,
                    attempts: 0,
                    duration_ms: None,
                },
            ],
            ..Default::default()
        };

        assert_eq!(shell_prompt(&session), "› [2/3 running (coder)] ");
    }

    #[test]
    fn format_results_lines_shows_outputs() {
        let session = CommanderSession {
            worker_outputs: vec![
                ("planner".to_string(), "Plan output".to_string()),
                ("reviewer".to_string(), "Review output".to_string()),
            ],
            ..Default::default()
        };

        assert_eq!(
            format_results_lines(&session),
            vec![
                "Completed worker outputs:".to_string(),
                "\n[planner]".to_string(),
                "Plan output".to_string(),
                "\n[reviewer]".to_string(),
                "Review output".to_string(),
            ]
        );
    }

    #[test]
    fn execution_workers_with_auto_reviewer_appends_when_missing() {
        let workers = vec![WorkerDefinition {
            name: "coder".to_string(),
            instructions: "Implement changes".to_string(),
        }];

        let cfg = CommanderRunConfig::default();
        let execution_workers = execution_workers_with_auto_workers(&workers, &cfg);

        assert_eq!(execution_workers.len(), 2);
        assert_eq!(execution_workers[1].name, "reviewer");
    }

    #[test]
    fn execution_workers_with_auto_reviewer_does_not_add_for_empty_workers() {
        let cfg = CommanderRunConfig::default();
        let execution_workers = execution_workers_with_auto_workers(&[], &cfg);
        assert!(execution_workers.is_empty());
    }

    #[test]
    fn execution_workers_with_auto_reviewer_keeps_existing_reviewer() {
        let workers = vec![
            WorkerDefinition {
                name: "coder".to_string(),
                instructions: "Implement changes".to_string(),
            },
            WorkerDefinition {
                name: "reviewer".to_string(),
                instructions: "Custom review".to_string(),
            },
        ];

        let cfg = CommanderRunConfig::default();
        let execution_workers = execution_workers_with_auto_workers(&workers, &cfg);

        assert_eq!(execution_workers.len(), 2);
        assert_eq!(execution_workers[1].instructions, "Custom review");
    }

    #[test]
    fn save_and_load_session_roundtrip() {
        let mut session = CommanderSession::default();
        session.task = Some("do the thing".to_string());
        session.workers = vec![WorkerDefinition {
            name: "planner".to_string(),
            instructions: "plan it".to_string(),
        }];

        let file = NamedTempFile::new().expect("tmp");
        save_session(&session, file.path()).expect("save");

        let mut loaded = CommanderSession::default();
        load_session_into(&mut loaded, file.path()).expect("load");

        assert_eq!(loaded.task, Some("do the thing".to_string()));
        assert_eq!(
            loaded.workers,
            vec![WorkerDefinition {
                name: "planner".to_string(),
                instructions: "plan it".to_string()
            }]
        );
    }
}
