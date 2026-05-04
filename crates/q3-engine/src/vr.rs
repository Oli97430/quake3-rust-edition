//! Support VR (OpenXR) — scaffold.
//!
//! # État actuel
//!
//! **Scaffold.**  L'infrastructure pour passer le renderer en rendu
//! stéréo + récupérer les poses HMD/controllers est ébauchée ici, mais
//! la VR n'est pas fonctionnelle de bout en bout :
//!
//! - L'intégration OpenXR réelle demande une dépendance `openxr` + la
//!   création d'une session wgpu partagée avec un swapchain VK/D3D12
//!   fourni par le runtime.  Ce scaffold *n'ajoute pas la dép* pour
//!   ne pas alourdir le workspace tant que la VR n'est pas prioritaire ;
//!   ajouter `openxr = "0.19"` et retirer les `cfg(feature = "vr")`
//!   autour de `#[allow(unused)]` activera le chemin.
//! - Le rendu stéréo nécessite de faire deux passes monde avec les deux
//!   matrices view/projection retournées par OpenXR.  `q3-renderer` est
//!   aujourd'hui mono-caméra ; un refactor camera serait à faire.
//! - Les controllers remplaceront clavier/souris pour viser+tirer : hook
//!   à cabler vers `Input` (voir `app.rs`).
//!
//! # Modèle de threading
//!
//! OpenXR a son propre `xrWaitFrame` / `xrBeginFrame` / `xrEndFrame`
//! qui doit encadrer le rendu.  L'idée est :
//!
//! ```text
//!   loop:
//!     xrWaitFrame         → temps de rendu cible
//!     poll controllers    → usercmd local
//!     tick simulation
//!     for eye in [L, R]:
//!         xrAcquireSwapchainImage
//!         render monde (view/proj de l'œil)
//!         xrReleaseSwapchainImage
//!     xrEndFrame
//! ```
//!
//! On peut désactiver la VR en runtime via `--no-vr` ou l'absence de
//! runtime OpenXR détecté — auquel cas on retombe sur le rendu mono
//! classique sans rien casser.

// Scaffolding VR — l'intégration OpenXR runtime arrivera dans une
// release dédiée. Tous les types/fields en `dead_code` ici sont
// l'API publique qui sera consommée à ce moment-là. On évite de
// les retirer pour ne pas re-bricoler la struct deux fois.
#![allow(dead_code)]

use q3_math::{Mat4, Vec3};

/// Mode d'affichage du renderer.  `Mono` = un seul viewport ; `Stereo`
/// = deux viewports (un par œil), avec poses de tête séparées.
#[derive(Debug, Clone, Copy, Default)]
pub enum DisplayMode {
    #[default]
    Mono,
    Stereo,
}

/// Pose stéréo d'une frame VR : matrice view + matrice proj pour
/// chaque œil, fournies par OpenXR.  En mono, seul `eye[0]` est utilisé
/// et `Mat4::IDENTITY` occupe `eye[1]`.
#[derive(Debug, Clone, Copy)]
pub struct StereoPose {
    pub view: [Mat4; 2],
    pub proj: [Mat4; 2],
    /// Position monde de la tête (milieu des deux yeux) — pour le son 3D
    /// et la sélection du near plane du viewmodel.
    pub head_position: Vec3,
}

impl Default for StereoPose {
    fn default() -> Self {
        Self {
            view: [Mat4::IDENTITY; 2],
            proj: [Mat4::IDENTITY; 2],
            head_position: Vec3::ZERO,
        }
    }
}

/// État controller VR — 1 par main.  Position/orientation monde + axes
/// d'input (trigger, stick, grip).  Mappés ensuite vers `Input::forward_axis`
/// etc. côté app.
#[derive(Debug, Clone, Copy, Default)]
pub struct ControllerState {
    pub pose: Mat4,
    pub trigger: f32, // [0..1]
    pub grip: f32,
    pub stick: [f32; 2], // [x=-1..1 left/right, y=-1..1 down/up]
    pub primary_button: bool,
    pub secondary_button: bool,
}

/// Runtime VR côté engine.  Actuellement un stub — les méthodes
/// retournent les valeurs par défaut, mais la forme est là pour qu'un
/// futur commit branche OpenXR sans toucher à `app.rs`.
pub struct VrRuntime {
    enabled: bool,
    pose: StereoPose,
    left: ControllerState,
    right: ControllerState,
}

impl VrRuntime {
    /// Tente d'initialiser OpenXR.  Retourne un runtime "disabled" si
    /// aucun runtime XR n'est détecté ou si `--no-vr` est passé en CLI
    /// (typiquement : PC non-VR, CI, dev headless).
    pub fn init(enable_request: bool) -> Self {
        if !enable_request {
            tracing::debug!("vr: désactivé (pas de --vr)");
            return Self::disabled();
        }
        // TODO(vr) : ici, `openxr::Entry::linked()` puis
        // `entry.create_instance(...)` avec l'extension wgpu adaptée
        // (`KHR_vulkan_enable2` sur Vulkan, `KHR_D3D12_enable` sur
        // DX12).  Si ça échoue → fallback.
        tracing::warn!(
            "vr: scaffold seulement — OpenXR non branché, rendu mono conservé"
        );
        Self::disabled()
    }

    fn disabled() -> Self {
        Self {
            enabled: false,
            pose: StereoPose::default(),
            left: ControllerState::default(),
            right: ControllerState::default(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn display_mode(&self) -> DisplayMode {
        if self.enabled {
            DisplayMode::Stereo
        } else {
            DisplayMode::Mono
        }
    }

    /// Appelé en début de frame — poll `xrWaitFrame` + poll des
    /// controllers.  En mode disabled : no-op.
    pub fn begin_frame(&mut self) {
        if !self.enabled {
            return;
        }
        // TODO(vr) : xrWaitFrame + sync action set + get_view_locate.
    }

    pub fn pose(&self) -> StereoPose {
        self.pose
    }

    pub fn left(&self) -> ControllerState {
        self.left
    }

    pub fn right(&self) -> ControllerState {
        self.right
    }

    /// Fin de frame — submit les swapchain images. No-op en mode
    /// disabled.
    pub fn end_frame(&mut self) {
        if !self.enabled {
            return;
        }
        // TODO(vr) : xrEndFrame avec les 2 composition layers.
    }
}

impl Default for VrRuntime {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled() {
        let r = VrRuntime::default();
        assert!(!r.is_enabled());
        assert!(matches!(r.display_mode(), DisplayMode::Mono));
    }

    #[test]
    fn init_without_request_stays_disabled() {
        let r = VrRuntime::init(false);
        assert!(!r.is_enabled());
    }
}
