//! The selection outline (`overlay_outline.hh`, Viewport Overlays ▸ Objects
//! ▸ Outline Selected).
//!
//! Blender draws it in two passes. A prepass rasterizes the *selected*
//! objects alone into a 16-bit id buffer, each pixel holding the object's
//! colour class in the top two bits and the object itself below
//! (`outline_id_pack`, `overlay_outline_prepass_vert.glsl`); a screen pass
//! then finds the pixels whose neighbours hold another id and lays the line
//! there, in the theme colour the class names
//! (`overlay_outline_detect_frag.glsl`).
//!
//! The tracer does the same with what it already has. Every kernel leaves
//! the instance it met under each pixel (`util.wgsl`'s `outline_store`),
//! and this table says which class each instance is in — a click that
//! changes the selection uploads one word an object
//! ([`Tracer::outline`](super::Tracer::outline)) and the samples already
//! traced stand, where keeping the class on the instance would have meant
//! re-uploading the instance table and starting the image over.

/// An object's standing in the outline pass (`outline_colorid_get`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Outline {
    /// Not outlined: unselected, or the overlay is off.
    #[default]
    None,
    /// Being moved (`G_TRANSFORM_OBJ`): the theme's `transform`.
    Transform,
    /// Selected: the theme's `object_select`.
    Select,
    /// The active object: the theme's `active_object`.
    Active,
}

/// Where the colour class sits in a packed outline id (`outline_id_pack`).
pub const OUTLINE_SHIFT: u32 = 14;
/// The object's share of a packed id: 16K objects may be told apart.
pub const OUTLINE_MASK: u32 = (1 << OUTLINE_SHIFT) - 1;

impl Outline {
    /// Blender's colour id: 0 transform, 1 selected, 3 active — what the
    /// detect pass switches the theme colour on.
    pub fn color_id(self) -> u32 {
        match self {
            Self::None | Self::Transform => 0,
            Self::Select => 1,
            Self::Active => 3,
        }
    }

    /// The word the class table holds for an object: 0 when it takes no
    /// outline, else its colour id plus one — so a zeroed table, and a
    /// scene nothing is selected in, outlines nothing.
    pub fn word(self) -> u32 {
        match self {
            Self::None => 0,
            other => other.color_id() + 1,
        }
    }

    /// The class a table word stands for; anything else is [`Self::None`].
    pub fn of(word: u32) -> Self {
        match word {
            1 => Self::Transform,
            2 => Self::Select,
            4 => Self::Active,
            _ => Self::None,
        }
    }

    /// The id the prepass leaves at a pixel showing instance `inst`
    /// (`outline_id_pack`): 0 where the object takes no outline, which is
    /// also what the background holds.
    pub fn packed(self, inst: u32) -> u32 {
        match self {
            Self::None => 0,
            other => (other.color_id() << OUTLINE_SHIFT) | ((inst + 1) & OUTLINE_MASK),
        }
    }

    /// The class table for a scene whose instances are `classes`, in the
    /// order the scene was built in: what [`Tracer::outline`](super::Tracer::outline) takes.
    pub fn table(classes: &[Self]) -> Vec<u32> {
        classes.iter().map(|c| c.word()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word round-trips, and the three classes keep Blender's colour ids.
    #[test]
    fn a_class_survives_the_table_word() {
        for class in [Outline::None, Outline::Transform, Outline::Select, Outline::Active] {
            assert_eq!(Outline::of(class.word()), class, "{class:?}");
        }
        assert_eq!((Outline::Transform.color_id(), Outline::Select.color_id(), Outline::Active.color_id()), (0, 1, 3));
        assert_eq!(Outline::table(&[Outline::Select, Outline::None, Outline::Active]), vec![2, 0, 4]);
    }

    /// The packed id holds the class above the object, and nothing at all
    /// for an object that takes no outline — the background's own value, so
    /// the detect pass needs no second test for it.
    #[test]
    fn the_packed_id_holds_the_class_above_the_object() {
        // Instance 0 selected: class 1 up top, object 1 below.
        assert_eq!(Outline::Select.packed(0), (1 << 14) | 1);
        assert_eq!(Outline::Active.packed(6), (3 << 14) | 7);
        // Transform's class is 0, so only the object tells it from nothing.
        assert_eq!(Outline::Transform.packed(0), 1);
        assert_eq!(Outline::None.packed(6), 0);
        // Past 16K objects the id wraps, as Blender's 14 bits do.
        assert_eq!(Outline::Select.packed(OUTLINE_MASK), 1 << 14);
    }
}
