//! Content flags et surface flags — fidèles à `surfaceflags.h` de Q3.
//!
//! Les valeurs numériques sont identiques au C original pour rester
//! compatible avec les BSP compilés (les flags sont stockés dans le fichier).

use bitflags::bitflags;

bitflags! {
    /// Qu'est-ce qui *remplit* un brush ? Solide, eau, lava, teleporter…
    ///
    /// Un trace passe `mask: Contents` — seul un brush dont le contenu
    /// recoupe le masque est considéré comme obstacle.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct Contents: u32 {
        const SOLID           = 0x0000_0001;
        const LAVA            = 0x0000_0008;
        const SLIME           = 0x0000_0010;
        const WATER           = 0x0000_0020;
        const FOG             = 0x0000_0040;
        const NOTTEAM1        = 0x0000_0080;
        const NOTTEAM2        = 0x0000_0100;
        const NOBOTCLIP       = 0x0000_0200;
        const AREAPORTAL      = 0x0000_8000;
        const PLAYERCLIP      = 0x0001_0000;
        const MONSTERCLIP     = 0x0002_0000;
        const TELEPORTER      = 0x0004_0000;
        const JUMPPAD         = 0x0008_0000;
        const CLUSTERPORTAL   = 0x0010_0000;
        const DONOTENTER      = 0x0020_0000;
        const BOTCLIP         = 0x0040_0000;
        const MOVER           = 0x0080_0000;
        const ORIGIN          = 0x0100_0000;
        const BODY            = 0x0200_0000;
        const CORPSE          = 0x0400_0000;
        const DETAIL          = 0x0800_0000;
        const STRUCTURAL      = 0x1000_0000;
        const TRANSLUCENT     = 0x2000_0000;
        const TRIGGER         = 0x4000_0000;
        const NODROP          = 0x8000_0000;

        /// Masque standard pour un joueur vivant.
        const MASK_PLAYERSOLID =
            Self::SOLID.bits() | Self::PLAYERCLIP.bits() | Self::BODY.bits();
        /// Masque pour un cadavre (ne s'arrête plus sur les clip-joueurs).
        const MASK_DEADSOLID = Self::SOLID.bits() | Self::PLAYERCLIP.bits();
        /// Masque pour un projectile (ignore les clip-joueurs).
        const MASK_SHOT =
            Self::SOLID.bits() | Self::BODY.bits() | Self::CORPSE.bits();
        /// Tout liquide.
        const MASK_WATER = Self::WATER.bits() | Self::LAVA.bits() | Self::SLIME.bits();
        /// Masque de tir opaque au solide + à la géométrie détaillée.
        const MASK_OPAQUE = Self::SOLID.bits() | Self::SLIME.bits() | Self::LAVA.bits();
    }
}

bitflags! {
    /// Propriétés de surface (son, marques, etc.). Stocké dans `DShader`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct SurfaceFlags: u32 {
        const NODAMAGE    = 0x0000_0001;
        const SLICK       = 0x0000_0002;
        const SKY         = 0x0000_0004;
        const LADDER      = 0x0000_0008;
        const NOIMPACT    = 0x0000_0010;
        const NOMARKS     = 0x0000_0020;
        const FLESH       = 0x0000_0040;
        const NODRAW      = 0x0000_0080;
        const HINT        = 0x0000_0100;
        const SKIP        = 0x0000_0200;
        const NOLIGHTMAP  = 0x0000_0400;
        const POINTLIGHT  = 0x0000_0800;
        const METALSTEPS  = 0x0000_1000;
        const NOSTEPS     = 0x0000_2000;
        const NONSOLID    = 0x0000_4000;
        const LIGHTFILTER = 0x0000_8000;
        const ALPHASHADOW = 0x0001_0000;
        const NODLIGHT    = 0x0002_0000;
        const DUST        = 0x0004_0000;
    }
}
