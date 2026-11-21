//! Shader hot reload: edit a `.wgsl` and see it without rebuilding.
//!
//! phiray bakes every shader with `include_str!`, which is right for
//! shipping — the WGSL travels in the binary and the wasm build has no
//! filesystem to read — but it means a one-character shader edit costs a
//! full rebuild of a crate pinned at `opt-level = 3`, about two minutes.
//! This is the developer's way round that, and it is **off unless asked
//! for**.
//!
//! Switch it on by pointing `PHIRAY_SHADERS` at the directory holding the
//! sources (`crates/phiray/src/trace/wgsl`), then call [`reload`] whenever
//! you want the files re-read — on a timer, or from a key. A kernel that
//! lists its [`Kernel::FILE_LIST`](crate::Kernel::FILE_LIST) is then assembled from
//! disk instead of from the baked text.
//!
//! # What it costs when it is off
//!
//! One relaxed atomic load per [`Kernel::key`] call, which is per dispatch.
//! Nothing is read, allocated or locked: [`generation`] returns 0 and
//! `key()` returns the same number it always did. That matters because the
//! key is taken on the hot path of every sample.
//!
//! # From the Apple tree
//!
//! The shape is `appkit/gpu/material/hot_reload/manager.rs`, with the three
//! ideas worth having from it (`docs/PORT_LEDGER.md` §3.5):
//!
//! 1. **Compare file CONTENTS, not mtime** — a save that changed nothing,
//!    or a `touch`, must not recompile.
//! 2. **A failed compile keeps the last good text.** Here the compile
//!    happens later, in `Gpu::pipeline`, so what this module guarantees is
//!    narrower and the same in spirit: a file that cannot be read leaves
//!    the previous text in place and records why ([`errors`]).
//! 3. **Budget the work and say when it is exceeded.**

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Longer than this for one `reload` and it says so (Apple's budget, which
/// was for a compile; this is only the read, so exceeding it means the
/// directory is the problem).
const BUDGET: Duration = Duration::from_secs(2);

/// Bumped by every [`reload`] that changed anything. `0` means hot reload
/// has never changed a file, which is the shipping case.
static GENERATION: AtomicU64 = AtomicU64::new(0);

struct Watched {
    /// The directory `PHIRAY_SHADERS` named.
    dir: std::path::PathBuf,
    /// File name to the text last read successfully.
    text: HashMap<String, String>,
    /// Why a file could not be read, most recent first.
    errors: Vec<String>,
}

fn watched() -> &'static Mutex<Option<Watched>> {
    static WATCHED: OnceLock<Mutex<Option<Watched>>> = OnceLock::new();
    WATCHED.get_or_init(|| {
        let dir = std::env::var_os("PHIRAY_SHADERS").map(std::path::PathBuf::from).filter(|d| d.is_dir());
        Mutex::new(dir.map(|dir| Watched { dir, text: HashMap::new(), errors: Vec::new() }))
    })
}

fn with<R>(f: impl FnOnce(&mut Option<Watched>) -> R) -> R {
    let mut guard = watched().lock().unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

/// Watch `dir` from here on, whatever the environment said.
///
/// The directory is read from `PHIRAY_SHADERS` once per process (a
/// `OnceLock`), which is right for a program but leaves the behaviour
/// untestable in-process — a test that sets the variable has already missed
/// the read. This is the seam: the tests drive the real [`reload`] and
/// [`assemble`] against a temporary directory through it.
#[cfg(test)]
pub(crate) fn watch_for_test(dir: std::path::PathBuf) {
    with(|slot| *slot = Some(Watched { dir, text: HashMap::new(), errors: Vec::new() }));
}

/// Whether `PHIRAY_SHADERS` named a directory that exists.
pub fn live() -> bool {
    with(|w| w.is_some())
}

/// How many times a reload has changed something. Folded into
/// [`Kernel::key`](crate::Kernel::key) so a changed file asks the pipeline
/// cache for a pipeline it has never compiled — which is the whole of the
/// invalidation, since the cache is keyed by that number.
pub fn generation() -> u64 {
    GENERATION.load(Ordering::Relaxed)
}

/// The text last read for `file`, when hot reload is live and has it.
pub fn text(file: &str) -> Option<String> {
    with(|w| w.as_ref()?.text.get(file).cloned())
}

/// Why files could not be read, from the last [`reload`].
pub fn errors() -> Vec<String> {
    with(|w| w.as_ref().map(|w| w.errors.clone()).unwrap_or_default())
}

/// Re-read every `.wgsl` in the watched directory. Returns the names whose
/// **contents** changed — a save that wrote the same bytes changes nothing
/// and recompiles nothing.
///
/// A file that cannot be read leaves whatever was there before it in place
/// and pushes the reason onto [`errors`], so a half-saved file never blanks
/// a kernel that was working.
pub fn reload() -> Vec<String> {
    let began = Instant::now();
    let changed = with(|slot| {
        let Some(w) = slot.as_mut() else { return Vec::new() };
        w.errors.clear();
        let Ok(entries) = std::fs::read_dir(&w.dir) else {
            w.errors.push(format!("cannot read {}", w.dir.display()));
            return Vec::new();
        };
        let mut changed = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_string) else { continue };
            match std::fs::read_to_string(&path) {
                // Contents, not mtime: identical bytes are not a change.
                Ok(next) if w.text.get(&name).is_some_and(|had| *had == next) => {}
                Ok(next) => {
                    w.text.insert(name.clone(), next);
                    changed.push(name);
                }
                Err(why) => w.errors.push(format!("{name}: {why}")),
            }
        }
        changed
    });
    // `eprintln!` rather than the `log` facade: phiray's dependency list is
    // deliberately short (bytemuck, wgpu, glam, rayon, pollster, web-time)
    // and this path only runs for a developer who set the variable.
    if !changed.is_empty() {
        GENERATION.fetch_add(1, Ordering::Relaxed);
        eprintln!("phiray: shaders hot-reloaded in {:?}: {}", began.elapsed(), changed.join(", "));
    }
    for why in with(|w| w.as_ref().map(|w| w.errors.clone()).unwrap_or_default()) {
        eprintln!("phiray: shader not read, keeping the last good text - {why}");
    }
    if began.elapsed() > BUDGET {
        eprintln!("phiray: shader reload took {:?}, over the {BUDGET:?} budget - is PHIRAY_SHADERS on a slow path?", began.elapsed());
    }
    changed
}

/// A kernel's source assembled from disk: its `files` in order, each read
/// from the watched directory. `None` unless hot reload is live and **every**
/// one of them is there — a kernel half from disk and half from the binary
/// would be neither.
pub(crate) fn assemble(files: &[&str]) -> Option<String> {
    if files.is_empty() || !live() {
        return None;
    }
    with(|w| {
        let w = w.as_ref()?;
        let mut out = String::new();
        for file in files {
            out.push_str(w.text.get(*file)?);
        }
        Some(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_is_off_and_free_without_the_variable() {
        // The test process has no `PHIRAY_SHADERS`, so the whole module is
        // inert: no directory, nothing assembled, and the generation that
        // `Kernel::key` folds in is zero.
        if std::env::var_os("PHIRAY_SHADERS").is_none() {
            assert!(!live(), "no variable, no watching");
            assert_eq!(generation(), 0, "the key is the one it always was");
            assert!(assemble(&["camera.wgsl"]).is_none());
            assert!(text("camera.wgsl").is_none());
            assert!(errors().is_empty());
            assert!(reload().is_empty(), "nothing to reload");
        }
    }

    #[test]
    fn a_kernel_that_lists_no_files_never_reads_from_disk() {
        // Opting in is per kernel: an empty `FILE_LIST` is the baked path, even
        // with hot reload live.
        assert!(assemble(&[]).is_none());
    }

    /// The whole loop, against a real directory: a file appears, `reload`
    /// reports it and bumps the generation, `assemble` returns it in the
    /// order the kernel asked for, a save of the SAME bytes changes nothing,
    /// and a changed byte is a change again.
    ///
    /// These run as one test because the watched directory is process-wide:
    /// two tests driving it would race.
    #[test]
    fn reload_reads_contents_and_only_counts_real_changes() {
        let dir = std::env::temp_dir().join(format!("phiray-hot-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("a temp dir");
        let write = |name: &str, text: &str| std::fs::write(dir.join(name), text).expect("write");

        write("a.wgsl", "// a\n");
        write("b.wgsl", "// b\n");
        // Not a shader: it must be ignored entirely.
        write("notes.txt", "ignore me");
        watch_for_test(dir.clone());
        assert!(live());

        let mut changed = reload();
        changed.sort();
        assert_eq!(changed, ["a.wgsl", "b.wgsl"], "both shaders, and not the .txt");
        let after_first = generation();
        assert!(after_first > 0, "a change bumps the generation");

        // Assembled in the order the kernel lists, not the directory's.
        assert_eq!(assemble(&["b.wgsl", "a.wgsl"]).as_deref(), Some("// b\n// a\n"));
        // A file the kernel wants but the directory lacks: all or nothing.
        assert!(assemble(&["a.wgsl", "missing.wgsl"]).is_none(), "half from disk is neither");

        // The same bytes written again: not a change, and the generation
        // holds, so no pipeline is thrown away.
        write("a.wgsl", "// a\n");
        assert!(reload().is_empty(), "identical contents are not a change");
        assert_eq!(generation(), after_first, "and nothing recompiles");

        // One byte different: a change.
        write("a.wgsl", "// a!\n");
        assert_eq!(reload(), ["a.wgsl"]);
        assert!(generation() > after_first);
        assert_eq!(assemble(&["a.wgsl"]).as_deref(), Some("// a!\n"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
