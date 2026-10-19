//! Where the process's one device lives.
//!
//! Two implementations, because `Gpu` is `Send` on a desktop and is not in a
//! browser. On wasm a `wgpu::Device` is a handle to a JavaScript object —
//! `wgpu::backend::webgpu::WebDevice` holds an `Rc<Cell<u32>>` — and a JS
//! object belongs to the thread that made it. So:
//!
//! * A `static Mutex<Option<Option<Gpu>>>` needs `Sync`, which needs `Send`,
//!   which the web device does not have. The static does not compile there
//!   at all: *"`Rc<Cell<u32>>` cannot be sent between threads safely"*.
//! * A `thread_local!` asks for neither, and says the true thing: the device
//!   belongs to this thread. On the web there is one thread anyway
//!   (`gpui_web` without `multithreaded`), so nothing is lost.
//!
//! [`with`] and [`try_with`] hide the difference, so `Gpu::shared`,
//! `Gpu::switch` and `Gpu::opened` each keep one body. The `Option<Option<_>>`
//! is the same on both: outer `None` means "not opened yet", inner `None`
//! means "opened and there was no adapter".

use super::Gpu;

type Slot = Option<Option<Gpu>>;

#[cfg(not(target_family = "wasm"))]
static SHARED: std::sync::Mutex<Slot> = std::sync::Mutex::new(None);

#[cfg(target_family = "wasm")]
thread_local! {
    static SHARED: std::cell::RefCell<Slot> = const { std::cell::RefCell::new(None) };
}

/// Run `f` on the slot, waiting for it if another holder has it.
///
/// `None` when the slot cannot be reached: a poisoned mutex on a desktop, or
/// a re-entrant borrow in a browser — both meaning some other frame of this
/// call stack is already inside, which is a caller's bug, not a reason to
/// panic in a render loop.
pub(super) fn with<R>(f: impl FnOnce(&mut Slot) -> R) -> Option<R> {
    #[cfg(not(target_family = "wasm"))]
    {
        SHARED.lock().ok().map(|mut slot| f(&mut slot))
    }
    #[cfg(target_family = "wasm")]
    {
        SHARED.with(|cell| cell.try_borrow_mut().ok().map(|mut slot| f(&mut slot)))
    }
}

/// [`with`], but give up rather than wait — for [`Gpu::opened`], which asks
/// whether the device is open and must not block the thread that is opening
/// it.
pub(super) fn try_with<R>(f: impl FnOnce(&Slot) -> R) -> Option<R> {
    #[cfg(not(target_family = "wasm"))]
    {
        SHARED.try_lock().ok().map(|slot| f(&slot))
    }
    #[cfg(target_family = "wasm")]
    {
        SHARED.with(|cell| cell.try_borrow().ok().map(|slot| f(&slot)))
    }
}
