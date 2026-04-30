//! Caméra FPS — position, angles (pitch/yaw), et matrices view/proj.
//!
//! Les conventions Q3 (Z-up, X-forward) sont traduites vers le clip space
//! de wgpu (Y-up, Z vers l'écran, -1..1 en X/Y, 0..1 en Z).

use bytemuck::{Pod, Zeroable};
use q3_math::{Angles, Mat4, Vec3, Vec4};

/// Valeurs par défaut proches du jeu original (cg_fov 90, r_znear 4).
///
/// Q3 historique : `cg_fov` est le **FOV horizontal en 4:3**.  Pour
/// supporter proprement le 16:9, 21:9, 32:9 sans déformer l'image
/// verticale, on conserve cette sémantique : on stocke `fov_h_at_4_3`
/// et on dérive le FOV vertical par `vfov = 2·atan(tan(hfov/2) / (4/3))`.
/// Le `proj_matrix` ré-applique l'aspect courant sur ce vfov, ce qui
/// élargit l'image **horizontalement** sur ultra-wide tout en gardant la
/// même altitude visible : c'est le scaling « Hor+ » attendu par les
/// joueurs modernes (les éléments verticaux ne sont pas tronqués sur
/// 32:9, le champ s'élargit côtés gauche/droit).
const DEFAULT_FOV_H_4_3_DEG: f32 = 90.0;
const DEFAULT_Z_NEAR: f32 = 4.0;
const DEFAULT_Z_FAR: f32 = 16384.0;
/// Référence d'aspect pour interpréter `cg_fov` comme un FOV horizontal.
const FOV_REF_ASPECT: f32 = 4.0 / 3.0;

/// Uniforme caméra passé au shader.
///
/// Layout aligné sur WGSL (std140-like) :
/// * `view_proj` : proj × view, utilisé par tout le rendu.
/// * `view_pos`  : position monde de la caméra + 1 pour padding.
/// * `inv_view_proj_rot` : inverse de (proj × view_sans_translation), pour
///   reconstituer une direction monde à partir d'un NDC — utile au skybox.
/// * `time_info` : `[time, 0, 0, 0]` — horloge applicative en secondes,
///   utilisée par les shaders qui animent les UV (`tcmod scroll/rotate`)
///   ou la couleur (`rgbgen wave`).  Les shaders qui n'en ont pas besoin
///   peuvent simplement ne pas déclarer ce champ dans leur struct WGSL.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view_pos: [f32; 4],
    pub inv_view_proj_rot: [[f32; 4]; 4],
    pub time_info: [f32; 4],
}

/// Caméra FPS style Quake.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: Vec3,
    pub angles: Angles,
    /// FOV horizontal **mesuré à 4:3** (sémantique Q3 `cg_fov`).
    /// Le FOV vertical effectif est dérivé via `fov_y_deg()`, ce qui
    /// permet au moteur d'élargir le champ horizontal sur ultra-wide
    /// sans que le joueur paramètre une 2e cvar.
    pub fov_h_at_4_3_deg: f32,
    pub aspect: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    pub fn new(position: Vec3, angles: Angles, aspect: f32) -> Self {
        Self {
            position,
            angles,
            fov_h_at_4_3_deg: DEFAULT_FOV_H_4_3_DEG,
            aspect,
            z_near: DEFAULT_Z_NEAR,
            z_far: DEFAULT_Z_FAR,
        }
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    /// Définit le FOV horizontal nominal (sémantique `cg_fov` Q3 : la
    /// valeur que le joueur règle dans les options, mesurée à 4:3).
    /// Clampé sur `[60, 130]` comme le jeu original pour bloquer les
    /// fish-eye absurdes / les zoom-snipe en dehors du game-design.
    pub fn set_horizontal_fov_4_3(&mut self, deg: f32) {
        self.fov_h_at_4_3_deg = deg.clamp(60.0, 130.0);
    }

    /// FOV vertical effectif en degrés. Dérivé : `vfov = 2·atan(tan(hfov/2) / (4/3))`.
    /// Indépendant de l'aspect courant — c'est `proj_matrix` qui élargit
    /// ensuite horizontalement selon `aspect`.
    pub fn fov_y_deg(&self) -> f32 {
        let half_h = (self.fov_h_at_4_3_deg.to_radians() * 0.5).tan();
        let half_v = half_h / FOV_REF_ASPECT;
        2.0 * half_v.atan().to_degrees()
    }

    /// FOV horizontal **effectif** sur l'aspect courant — dérivé via le
    /// vfov puis re-multiplié par l'aspect réel. Sur 16:9 ≈ 106°,
    /// sur 21:9 ≈ 120°, sur 32:9 ≈ 141°. Exposé pour debug / HUD.
    pub fn fov_x_effective_deg(&self) -> f32 {
        let half_y = (self.fov_y_deg().to_radians() * 0.5).tan();
        let half_x = half_y * self.aspect.max(0.0001);
        2.0 * half_x.atan().to_degrees()
    }

    /// Matrice de conversion Q3 (X-forward, Y-left, Z-up) → GL/wgpu
    /// (X-right, Y-up, -Z-forward).
    fn q3_to_gl() -> Mat4 {
        Mat4::from_cols(
            Vec4::new(0.0, 0.0, -1.0, 0.0), // Q3 X-forward → GL -Z
            Vec4::new(-1.0, 0.0, 0.0, 0.0), // Q3 Y-left    → GL -X
            Vec4::new(0.0, 1.0, 0.0, 0.0),  // Q3 Z-up      → GL +Y
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
    }

    pub fn view_matrix(&self) -> Mat4 {
        let basis = self.angles.to_vectors();
        // Construit une matrice "look along" : on regarde `basis.forward`.
        // Convention Q3 caméra : axe X = forward, axe Y = LEFT, axe Z = up
        // (right-handed, Z-up, X-forward). `q3_to_gl()` mappe ensuite Q3 Y
        // (left) → GL -X (right), etc. La colonne 1 de `cam_to_world` doit
        // donc être la direction « gauche » en world, soit `-basis.right`
        // (puisque `basis.right` est la droite world). Avant correctif on
        // passait `basis.right` tel quel → l'écran était miroir-flip en X,
        // donnant une sensation de souris/scène inversée.
        let cam_to_world = Mat4::from_cols(
            basis.forward.extend(0.0),
            (-basis.right).extend(0.0),
            basis.up.extend(0.0),
            self.position.extend(1.0),
        );
        Self::q3_to_gl() * cam_to_world.inverse()
    }

    pub fn proj_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(
            self.fov_y_deg().to_radians(),
            self.aspect.max(0.0001),
            self.z_near,
            self.z_far,
        )
    }

    pub fn view_proj(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// Matrice view privée de sa translation — ne garde que la rotation
    /// Q3→GL et l'orientation caméra. Utile pour un skybox qui ne doit pas
    /// avoir de parallaxe.
    pub fn view_matrix_rot_only(&self) -> Mat4 {
        let basis = self.angles.to_vectors();
        // Même convention que `view_matrix` : la colonne 1 est la direction
        // LEFT en world (`-basis.right`), pour matcher `q3_to_gl()`.
        let cam_to_world = Mat4::from_cols(
            basis.forward.extend(0.0),
            (-basis.right).extend(0.0),
            basis.up.extend(0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );
        Self::q3_to_gl() * cam_to_world.inverse()
    }

    /// Inverse de (proj × view_sans_translation) — passé en uniforme
    /// pour que le shader de skybox reconstitue une direction monde.
    pub fn inv_view_proj_rot(&self) -> Mat4 {
        (self.proj_matrix() * self.view_matrix_rot_only()).inverse()
    }

    /// Construit l'uniforme.  `time` (secondes applicatives) est injecté
    /// dans `time_info.x` — les shaders qui animent UV / couleur s'en
    /// servent ; les autres ignorent le champ.
    pub fn uniform(&self, time: f32) -> CameraUniform {
        CameraUniform {
            view_proj: self.view_proj().to_cols_array_2d(),
            view_pos: [self.position.x, self.position.y, self.position.z, 1.0],
            inv_view_proj_rot: self.inv_view_proj_rot().to_cols_array_2d(),
            time_info: [time, 0.0, 0.0, 0.0],
        }
    }

    /// Déplacement en coordonnées locales (forward/right/up = composantes).
    pub fn translate_local(&mut self, local: Vec3) {
        let b = self.angles.to_vectors();
        self.position += b.forward * local.x + b.right * local.y + b.up * local.z;
    }

    /// Ajoute des deltas d'angles (en degrés). Clamp le pitch à ±89°.
    pub fn rotate(&mut self, d_pitch: f32, d_yaw: f32) {
        self.angles.pitch = (self.angles.pitch + d_pitch).clamp(-89.0, 89.0);
        self.angles.yaw = q3_math::normalize_180(self.angles.yaw + d_yaw);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ancrage : un point situé à la DROITE du joueur (en coordonnées Q3,
    /// donc à -Y dans le world) doit tomber côté droit de l'écran en GL
    /// (clip_x positif après view·proj). On teste directement la view
    /// matrix car la proj RH conserve le signe de X en NDC.
    #[test]
    fn view_matrix_preserves_right_handedness() {
        let cam = Camera::new(
            Vec3::ZERO,
            Angles::new(0.0, 0.0, 0.0),
            16.0 / 9.0,
        );
        // 100 ahead (+X Q3), 100 to Q3-right (-Y), 50 up (+Z).
        let p_world = Vec4::new(100.0, -100.0, 50.0, 1.0);
        let p_view = cam.view_matrix() * p_world;
        assert!(
            p_view.x > 0.0,
            "point à droite du joueur attendu côté GL +X (clip_x>0), obtenu {}",
            p_view.x
        );
        assert!(
            p_view.z < 0.0,
            "point devant le joueur attendu GL -Z, obtenu {}",
            p_view.z
        );
        assert!(
            p_view.y > 0.0,
            "point au-dessus du joueur attendu GL +Y, obtenu {}",
            p_view.y
        );
    }

    /// `cg_fov = 90°` mesuré à 4:3 doit produire un vfov ≈ 73.7°,
    /// indépendant de l'aspect — l'aspect n'affecte que `proj_matrix`
    /// (scaling horizontal), pas la sémantique de la cvar.
    #[test]
    fn vfov_derives_from_hfov_at_4_3_independent_of_actual_aspect() {
        for &aspect in &[4.0 / 3.0, 16.0 / 9.0, 21.0 / 9.0, 32.0 / 9.0] {
            let cam = Camera::new(Vec3::ZERO, Angles::new(0.0, 0.0, 0.0), aspect);
            let vfov = cam.fov_y_deg();
            assert!(
                (vfov - 73.74).abs() < 0.05,
                "vfov pour cg_fov=90 attendu ≈ 73.74°, obtenu {vfov} (aspect={aspect})"
            );
        }
    }

    /// Le FOV horizontal **effectif** doit s'élargir avec l'aspect — c'est
    /// le contrat Hor+ : sur 32:9 le joueur voit plus à gauche/droite
    /// qu'en 4:3, mais la même tranche verticale.  Ancrage des valeurs :
    /// 4:3 → 90° (par construction), 16:9 → ~106°, 21:9 → ~120°,
    /// 32:9 → ~141°.  Tolérances larges parce que les ratios "21:9"
    /// commerciaux sont en réalité 64:27 ≈ 2.37, etc.
    #[test]
    fn effective_hfov_widens_with_aspect_ratio() {
        let make = |aspect| {
            let cam = Camera::new(Vec3::ZERO, Angles::new(0.0, 0.0, 0.0), aspect);
            cam.fov_x_effective_deg()
        };
        assert!((make(4.0 / 3.0) - 90.0).abs() < 0.1, "4:3 attendu 90°");
        let h_169 = make(16.0 / 9.0);
        let h_219 = make(21.0 / 9.0);
        let h_329 = make(32.0 / 9.0);
        assert!(h_169 > 100.0 && h_169 < 110.0, "16:9 hfov hors fenêtre : {h_169}");
        assert!(h_219 > 115.0 && h_219 < 125.0, "21:9 hfov hors fenêtre : {h_219}");
        assert!(h_329 > 135.0 && h_329 < 145.0, "32:9 hfov hors fenêtre : {h_329}");
        // Strictement croissant : aucune plateau, aucune régression
        // pour un aspect intermédiaire bizarre.
        assert!(h_169 < h_219 && h_219 < h_329, "hfov pas monotone");
    }

    /// La projection `Mat4::perspective_rh` doit rester valide (matrice
    /// inversible) sur un éventail extrême d'aspects, sans NaN.  C'est le
    /// garde-fou qui assure qu'un `Resized` 32:9 ne casse pas le rendu.
    #[test]
    fn proj_matrix_finite_across_extreme_aspects() {
        for &aspect in &[1.0, 4.0 / 3.0, 16.0 / 10.0, 16.0 / 9.0, 21.0 / 9.0, 32.0 / 9.0, 48.0 / 9.0] {
            let cam = Camera::new(Vec3::ZERO, Angles::new(0.0, 0.0, 0.0), aspect);
            let m = cam.proj_matrix().to_cols_array();
            for v in m {
                assert!(v.is_finite(), "proj contient NaN/inf à aspect={aspect}");
            }
        }
    }

    /// `set_horizontal_fov_4_3` doit clamper hors `[60, 130]` — bloque
    /// le quake-pro fish-eye à 170° et le zoom-snipe à 30° utilisés
    /// historiquement pour des avantages compétitifs.
    #[test]
    fn fov_clamped_into_q3_range() {
        let mut cam = Camera::new(Vec3::ZERO, Angles::new(0.0, 0.0, 0.0), 16.0 / 9.0);
        cam.set_horizontal_fov_4_3(30.0);
        assert_eq!(cam.fov_h_at_4_3_deg, 60.0);
        cam.set_horizontal_fov_4_3(170.0);
        assert_eq!(cam.fov_h_at_4_3_deg, 130.0);
        cam.set_horizontal_fov_4_3(95.0);
        assert_eq!(cam.fov_h_at_4_3_deg, 95.0);
    }

    /// Tourner yaw de +90° (Q3 : face +Y, gauche monde) doit transformer
    /// un point qui était à droite en un point... derrière la caméra
    /// (il était à 90° de notre droite avant, il est à 180° = derrière).
    #[test]
    fn view_matrix_rotates_consistently_with_yaw() {
        let cam = Camera::new(
            Vec3::ZERO,
            Angles::new(0.0, 90.0, 0.0),
            16.0 / 9.0,
        );
        let p_world = Vec4::new(100.0, -100.0, 0.0, 1.0);
        let p_view = cam.view_matrix() * p_world;
        // Après rotation, le point doit être derrière (GL +Z = derrière la caméra).
        assert!(
            p_view.z > 0.0,
            "après yaw +90°, un point situé à l'ex-droite doit être derrière, GL z = {}",
            p_view.z
        );
    }
}
