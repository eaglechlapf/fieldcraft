//! The scalar of a sparse matrix: `f64` for the real factorizations and `Complex64` for the
//! complex-symmetric ones of a harmonic response (K − ω²M + iωC, with K, M and C real).
//!
//! The matrices are **complex symmetric**, A = Aᵀ, never Hermitian: every product, solve,
//! factorization and residual transposes and never conjugates. The bilinear form xᵀ y is the
//! one the factorization preserves; a conjugate would change the answer for any complex
//! damping. The trait promises only what `f64` and `Complex64` both have: arithmetic, zero and
//! one, a modulus for the pivot tests, and finiteness. It has no order and no sign, so the real
//! inertia (Sylvester's law, [`crate::ldlt::Inertia`]) is defined for `f64` alone.

use std::cell::RefCell;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_complex::Complex64;

/// Rows packed together for the micro-tiles of the dense kernels ([`Scalar::with_packed`]).
pub const PACKED_ROWS: usize = 8;

/// An element of a sparse symmetric matrix and of the vectors it acts on.
pub trait Scalar:
    Copy
    + Send
    + Sync
    + std::fmt::Debug
    + Default
    + Sum
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    const ZERO: Self;
    const ONE: Self;

    /// |x|: the pivot tests compare moduli. A NaN stays NaN.
    fn modulus(self) -> f64;

    /// Both parts finite (every part of a complex value).
    fn is_finite(self) -> bool;

    /// Run `f` on a buffer of packed rows that the calling thread keeps from one call to the
    /// next, so a dense kernel's tiles take no memory of their own for them; on a new one while
    /// the thread's is in use. What is in the buffer when `f` gets it is of no meaning.
    fn with_packed<R>(f: impl FnOnce(&mut Vec<[Self; PACKED_ROWS]>) -> R) -> R;
}

impl Scalar for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;

    #[inline]
    fn modulus(self) -> f64 {
        self.abs()
    }

    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    fn with_packed<R>(f: impl FnOnce(&mut Vec<[f64; PACKED_ROWS]>) -> R) -> R {
        thread_local! {
            static PACKED: RefCell<Vec<[f64; PACKED_ROWS]>> = const { RefCell::new(Vec::new()) };
        }
        PACKED.with(|kept| match kept.try_borrow_mut() {
            Ok(mut rows) => f(&mut rows),
            Err(_) => f(&mut Vec::new()),
        })
    }
}

impl Scalar for Complex64 {
    const ZERO: Complex64 = Complex64::new(0.0, 0.0);
    const ONE: Complex64 = Complex64::new(1.0, 0.0);

    #[inline]
    fn modulus(self) -> f64 {
        self.norm()
    }

    #[inline]
    fn is_finite(self) -> bool {
        Complex64::is_finite(self)
    }

    fn with_packed<R>(f: impl FnOnce(&mut Vec<[Complex64; PACKED_ROWS]>) -> R) -> R {
        thread_local! {
            static PACKED: RefCell<Vec<[Complex64; PACKED_ROWS]>> = const { RefCell::new(Vec::new()) };
        }
        PACKED.with(|kept| match kept.try_borrow_mut() {
            Ok(mut rows) => f(&mut rows),
            Err(_) => f(&mut Vec::new()),
        })
    }
}
