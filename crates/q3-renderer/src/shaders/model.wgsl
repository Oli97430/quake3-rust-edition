// Rendu MD3 animé — interpolation linéaire entre deux frames.
//
// Le vertex shader reçoit (pos_a, normal_a) depuis le slot 0 et
// (pos_b, normal_b) depuis le slot 1, UV depuis le slot 2 (constant).
// `inst.lerp_pad.x` donne le coefficient de mélange [0..1].
// `inst.lerp_pad.y` est un flag viewmodel : 1.0 → arme tenue en main,
// shading renforcé (fresnel rim + specular) ; 0.0 → bot/world, shading
// "entity lighting" classique.
//
// Shading "lit flat" à partir de la normale interpolée et de la couleur de
// base échantillonnée dans la texture diffuse du shader. Pas de lightmap
// pour les modèles (ils sont éclairés par le light-grid dans le vrai Q3 ;
// on approxime par un N·L simple pour l'instant).

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct ModelUniform {
    model: mat4x4<f32>,
    color: vec4<f32>,
    lerp_pad: vec4<f32>,
};
@group(1) @binding(0) var<uniform> inst: ModelUniform;

@group(2) @binding(0) var diffuse_tex: texture_2d<f32>;
@group(2) @binding(1) var diffuse_smp: sampler;

struct VIn {
    @location(0) pos_a:    vec3<f32>,
    @location(1) normal_a: vec3<f32>,
    @location(2) pos_b:    vec3<f32>,
    @location(3) normal_b: vec3<f32>,
    @location(4) uv:       vec2<f32>,
};

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal_world: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    let t = inst.lerp_pad.x;
    let pos = mix(v.pos_a, v.pos_b, t);
    let n   = mix(v.normal_a, v.normal_b, t);
    let world = inst.model * vec4<f32>(pos, 1.0);
    out.clip_pos = camera.view_proj * world;
    // Normale : on ne re-normalise pas si la matrice est pure rotation×translation.
    out.normal_world = (inst.model * vec4<f32>(n, 0.0)).xyz;
    out.uv = v.uv;
    out.world_pos = world.xyz;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal_world);
    // Direction caméra → surface, normalisée.  Q3 = vers l'œil.
    let v = normalize(camera.view_pos.xyz - in.world_pos);

    // Éclairage type "entity lighting" Q3 : en Q3 original R_SetupEntityLighting
    // échantillonne le lightgrid et compose un ambient + un directed avec un
    // dot half-lambert pour lisser les zones d'ombre.  On n'a pas le lightgrid
    // côté renderer, donc on approxime :
    //
    // * une lumière clé en haut-avant-droit ;
    // * un fill opposé, plus faible, pour éviter des dos totalement noirs
    //   (c'est surtout visible sur les armes en 1re personne quand le joueur
    //   regarde une surface plein soleil et que l'arrière du flingue devient
    //   un aplat gris sale) ;
    // * un remap half-lambert (`n·l*0.5 + 0.5`) au lieu du `max(0, n·l)` qui
    //   crée une démarcation dure très peu flatteuse sur les models ronds.
    //
    // Résultat : range de luminosité [~0.55 .. ~1.10] — jamais plus sombre
    // que le fill, et avec un gradient doux sur toute la sphère directionelle.
    let key_dir  = normalize(vec3<f32>(0.35, 0.55, 0.85));
    let fill_dir = normalize(vec3<f32>(-0.35, -0.40, 0.30));
    let key  = dot(n, key_dir)  * 0.5 + 0.5; // [0, 1] half-lambert
    let fill = dot(n, fill_dir) * 0.5 + 0.5;
    let diff = 0.35 + key * 0.60 + fill * 0.15;

    let albedo = textureSample(diffuse_tex, diffuse_smp, in.uv);
    let base_rgb = albedo.rgb * inst.color.rgb * diff;

    // Viewmodel : shading enrichi.  On ajoute :
    //
    // * Une spéculaire Blinn-Phong sur la key_dir → reflets chromés qui
    //   donnent vie aux parties métalliques de l'arme (canon, culasse).
    //   Exposant élevé (shininess=48) pour un highlight serré, modulé par
    //   la luminance de l'albedo (parties brillantes = plus réfléchissantes,
    //   grip plastique mat = moins) — hack moins grossier que d'appliquer
    //   la même spéc partout.
    //
    // * Un rim-light fresnel (1-N·V)^3 → liseré lumineux sur les bords de
    //   la silhouette.  Utile pour détacher l'arme du fond sombre (caves,
    //   couloirs indoor) et générale à l'esthétique "hero asset" des
    //   shooters modernes.  Teinté légèrement plus chaud que blanc pur
    //   pour rester cohérent avec la key_dir diurne.
    //
    // Les deux termes sont additifs par-dessus le diffuse — ça peut passer
    // au-dessus de 1.0 en sRGB, mais wgpu fait un clamp implicite et le
    // résultat apparaît comme un highlight "sur-exposé" voulu.
    var final_rgb = base_rgb;
    if (inst.lerp_pad.y > 0.5) {
        // Half-vector pour Blinn-Phong.
        let h = normalize(key_dir + v);
        let nh = max(dot(n, h), 0.0);
        let spec_power = pow(nh, 48.0);
        // Modulation par luminance perceptuelle de l'albedo → le grip noir
        // ne scintille pas, mais le canon acier oui.
        let lum = dot(albedo.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        let spec_mask = smoothstep(0.25, 0.75, lum);
        let spec = spec_power * spec_mask * 0.8;
        let spec_col = vec3<f32>(1.0, 0.95, 0.85) * spec;

        // Rim light — fresnel Schlick-approx, puissance 3 pour limiter
        // l'épaisseur du liseré (sinon tout le modèle devient éclairé).
        let ndotv = clamp(dot(n, v), 0.0, 1.0);
        let fresnel = pow(1.0 - ndotv, 3.0);
        let rim_col = vec3<f32>(1.00, 0.90, 0.75) * (fresnel * 0.40);

        // Léger gain global pour que le viewmodel reste lisible même quand
        // la scène autour est très lumineuse — Q3 applique un `r_intensity`
        // similaire au viewmodel pour éviter qu'il disparaisse dans le HDR
        // (bien qu'en 1999 le "HDR" se limitait au sRGB saturé).
        final_rgb = base_rgb * 1.10 + spec_col + rim_col;
    }

    // Pas d'alpha-discard : la pipeline MD3 fait déjà du ALPHA_BLENDING.
    // Un seuil `a<0.5` rejetterait toute texture dont la stage sélectionnée
    // a blend additif (alpha=0 sur les pixels "éteints"), rendant muzzle
    // flashes et effets cyan totalement invisibles — ou des parties de
    // viewmodels si leur material pointe sur une pass blend-first.
    return vec4<f32>(final_rgb, max(albedo.a, inst.color.a));
}
