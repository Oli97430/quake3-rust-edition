//! Système audio Quake 3 basé sur **rodio**.
//!
//! Quake 3 gère jusqu'à 64 canaux simultanés avec priorités. Cette
//! implémentation reste fidèle à la logique (priority + channel reuse) mais
//! délègue le mixage et le spatialisation à rodio.
//!
//! # Features exposées
//!
//! * Chargement WAV et OGG depuis un buffer mémoire (via rodio::Decoder).
//! * Lecture spatialisée avec atténuation par distance.
//! * Musique de fond loopée, mute/volume globaux séparés.
//! * Listener (position + orientation) mis à jour par frame.
//!
//! # Améliorations vs C original
//!
//! * **Pas d'unsafe**, pas de globals mutables.
//! * **Sans OpenAL** : on sort par défaut sur `cpal` via rodio (Linux/Mac/Win
//!   et WASM).
//! * **Hot-reload** : le `SoundCache` est un `HashMap` externe, on peut
//!   `clear()` et recharger à chaud sans relancer le process.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

use hashbrown::HashMap;
use parking_lot::Mutex;
use q3_common::{Error, Result};
use q3_math::Vec3;
use rodio::{
    source::{SamplesConverter, Source},
    Decoder, OutputStream, OutputStreamHandle, Sink, SpatialSink,
};
use std::{io::Cursor, sync::Arc};
use tracing::{debug, warn};

/// Identifiant stable d'un sample chargé.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoundHandle(u32);

/// Identifiant d'un ambient loop attaché à une position monde (pour
/// `target_speaker` ou équivalent).  Utilisé pour arrêter la boucle
/// depuis l'appelant (map change, cvar mute, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopHandle(u32);

/// Priorité d'un son — plus haut = plus important, peut voler un canal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Ambient = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    Weapon = 4,
    /// Voix-off / annonces — jamais volées.
    VoiceOver = 5,
}

/// Paramètres d'un son spatialisé.
#[derive(Debug, Clone, Copy)]
pub struct Emitter3D {
    pub position: Vec3,
    /// Distance à laquelle le volume commence à chuter (unités Q3).
    pub near_dist: f32,
    /// Distance à laquelle le son devient silencieux.
    pub far_dist: f32,
    pub volume: f32,
    pub priority: Priority,
}

impl Default for Emitter3D {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            near_dist: 64.0,
            far_dist: 1024.0,
            volume: 1.0,
            priority: Priority::Normal,
        }
    }
}

/// Listener — typiquement suit la caméra du joueur.
#[derive(Debug, Clone, Copy, Default)]
pub struct Listener {
    pub position: Vec3,
    /// Axe droite de la tête (pour calculer la séparation stéréo).
    pub right: Vec3,
}

struct LoadedSound {
    /// Bytes encodés (WAV/OGG). On re-décode à chaque play pour pouvoir
    /// rejouer le même sample sans re-tamponner.
    bytes: Arc<[u8]>,
    #[allow(dead_code)]
    name: String,
}

struct Channel {
    sink: SpatialSink,
    priority: Priority,
}

/// Emetteur ambient en boucle continue — vit tant que la map est chargée
/// ou que l'appelant n'a pas demandé l'arrêt explicite.  On conserve
/// `emitter` pour recomputer l'atténuation à chaque `tick` au gré des
/// déplacements du listener.
struct LoopedEmitter {
    sink: SpatialSink,
    emitter: Emitter3D,
}

struct SoundState {
    sounds: HashMap<SoundHandle, LoadedSound>,
    next_handle: u32,
    channels: Vec<Option<Channel>>,
    listener: Listener,
    master_volume: f32,
    music_sink: Option<Sink>,
    music_volume: f32,
    /// Loops 3D actifs — clé = `LoopHandle` retourné par `play_3d_loop`.
    loops: HashMap<LoopHandle, LoopedEmitter>,
    next_loop: u32,
}

/// Façade thread-safe du système audio.
pub struct SoundSystem {
    _stream: OutputStream,
    handle: OutputStreamHandle,
    state: Mutex<SoundState>,
}

impl SoundSystem {
    /// Initialise rodio sur le périphérique par défaut.
    pub fn new() -> Result<Self> {
        Self::with_channels(64)
    }

    pub fn with_channels(max_channels: usize) -> Result<Self> {
        let (stream, handle) = OutputStream::try_default()
            .map_err(|e| Error::Renderer(format!("audio: init échouée ({e})")))?;
        let state = SoundState {
            sounds: HashMap::new(),
            next_handle: 1,
            channels: (0..max_channels).map(|_| None).collect(),
            listener: Listener::default(),
            master_volume: 1.0,
            music_sink: None,
            music_volume: 1.0,
            loops: HashMap::new(),
            next_loop: 1,
        };
        debug!("audio: rodio initialisé, {} canaux max", max_channels);
        Ok(Self {
            _stream: stream,
            handle,
            state: Mutex::new(state),
        })
    }

    /// Charge un sample WAV ou OGG et retourne un handle stable.
    pub fn load(&self, name: impl Into<String>, bytes: Vec<u8>) -> Result<SoundHandle> {
        let name: String = name.into();
        // On valide en essayant de décoder une fois.
        let cursor = Cursor::new(bytes.clone());
        Decoder::new(cursor)
            .map_err(|e| Error::Parse(format!("audio: décodage de '{}' échoué: {}", name, e)))?;

        let mut st = self.state.lock();
        let handle = SoundHandle(st.next_handle);
        st.next_handle = st.next_handle.wrapping_add(1).max(1);
        st.sounds.insert(
            handle,
            LoadedSound {
                bytes: bytes.into(),
                name,
            },
        );
        Ok(handle)
    }

    /// Libère un sample du cache.  À appeler explicitement sur les
    /// transitions de map pour éviter l'accumulation des samples en RAM
    /// (autrement le cache `sounds` grossissait sans bornes).
    pub fn unload(&self, handle: SoundHandle) -> bool {
        self.state.lock().sounds.remove(&handle).is_some()
    }

    /// Vide TOUT le cache de samples.  Utilisé sur changement de map
    /// pour éviter la fuite d'OOM en LAN sessions longues.  Les sons
    /// ré-utilisés ensuite seront ré-`load()` à la demande.
    pub fn clear_cache(&self) {
        let mut st = self.state.lock();
        let count = st.sounds.len();
        st.sounds.clear();
        if count > 0 {
            tracing::info!("audio: cache vidé ({count} samples libérés)");
        }
    }

    pub fn set_listener(&self, listener: Listener) {
        self.state.lock().listener = listener;
    }

    pub fn set_master_volume(&self, v: f32) {
        self.state.lock().master_volume = v.clamp(0.0, 1.0);
    }

    pub fn master_volume(&self) -> f32 {
        self.state.lock().master_volume
    }

    /// Joue un son spatialisé. Retourne `true` si un canal a pu être alloué.
    pub fn play_3d(&self, sound: SoundHandle, emitter: Emitter3D) -> bool {
        let mut st = self.state.lock();
        let Some(loaded) = st.sounds.get(&sound) else {
            warn!("audio: handle {:?} inconnu", sound);
            return false;
        };
        let bytes = loaded.bytes.clone();

        let dist = emitter.position.distance(st.listener.position);
        let atten = attenuation(dist, emitter.near_dist, emitter.far_dist);
        let volume = (emitter.volume * atten * st.master_volume).clamp(0.0, 1.0);
        if volume < 1e-3 {
            return false;
        }

        let slot = match Self::pick_channel(&mut st.channels, emitter.priority) {
            Some(i) => i,
            None => return false,
        };

        // **Optimisation alloc** (v0.9.5++ polish) — `bytes` est déjà
        // un `Arc<[u8]>` cloné depuis le cache (line 199) ; on l'enveloppe
        // directement dans un Cursor sans `to_vec()` qui allouait une
        // copie complète à chaque tir (samples WAV peuvent être 100+ KB).
        let Ok(decoder) = Decoder::new(Cursor::new(bytes)) else {
            warn!("audio: re-décodage échoué pour {:?}", sound);
            return false;
        };
        let source = decoder.convert_samples::<f32>();

        // **Relative coords pour SpatialSink** (v0.9.5++ fix) — rodio
        // applique sa propre atténuation interne basée sur la distance
        // entre `emitter_pos` et les `*_ear` positions.  En passant les
        // coords monde Q3 (souvent 10000+ unités), la distance internale
        // était énorme → atténuation rodio écrasait le volume à ≈0.
        // Solution : on translate tout dans le repère listener (listener
        // à l'origine, emitter à la position relative).  La distance
        // interne reflète alors la vraie distance perçue, et notre
        // `volume` calculé en amont (atténuation Q3 + master) prime.
        let right = st.listener.right;
        let ear_sep = 0.1;
        let left_ear = [-right.x * ear_sep, -right.y * ear_sep, -right.z * ear_sep];
        let right_ear = [right.x * ear_sep, right.y * ear_sep, right.z * ear_sep];
        let rel = emitter.position - st.listener.position;
        // On garde les unités Q3 mais autour de l'origine — rodio
        // intern atten ~ 1/dist. Pour une émission proche (eye = 64u
        // au-dessus du joueur) la distance reste 64, donc atten interne
        // ~1/64. C'est encore trop fort.  On scale par 1/100 pour que
        // les distances perçues soient en "mètres pseudo" (64u → 0.64m
        // → atten interne ~1).
        let scale = 0.01_f32;
        let emitter_pos = [rel.x * scale, rel.y * scale, rel.z * scale];

        let sink = match SpatialSink::try_new(&self.handle, emitter_pos, left_ear, right_ear) {
            Ok(s) => s,
            Err(e) => {
                warn!("audio: SpatialSink creation failed: {e}");
                return false;
            }
        };
        sink.set_volume(volume);
        sink.append(source);

        st.channels[slot] = Some(Channel {
            sink,
            priority: emitter.priority,
        });
        true
    }

    /// Démarre un son en boucle infinie à la position `emitter.position`.
    ///
    /// Utilisé pour les `target_speaker` de Q3 (bruits d'ambiance qui
    /// tournent tant que la map est chargée).  Contrairement à `play_3d`,
    /// on ne passe pas par le pool priorisé : chaque loop vit sur son
    /// propre `SpatialSink` et n'entre pas en compétition pour les 64
    /// canaux one-shot.  En contrepartie il faut explicitement `stop_loop`
    /// au changement de map, sinon la boucle continue.
    ///
    /// Retourne `None` si le handle est inconnu, si le décodeur échoue,
    /// ou si rodio ne peut pas créer de `SpatialSink` (OS audio en rade).
    /// Le volume et la position inter-ear sont recalculés à chaque `tick`
    /// pour suivre le listener : on n'a donc pas besoin de re-démarrer
    /// la boucle quand le joueur se déplace.
    pub fn play_3d_loop(&self, sound: SoundHandle, emitter: Emitter3D) -> Option<LoopHandle> {
        let mut st = self.state.lock();
        let loaded = st.sounds.get(&sound)?;
        let bytes = loaded.bytes.clone();

        // Décodage dédié — `SamplesConverter` est requis pour `.repeat_infinite()`.
        let decoder = Decoder::new(Cursor::new(bytes.as_ref().to_vec())).ok()?;
        let source: SamplesConverter<Decoder<Cursor<Vec<u8>>>, f32> =
            decoder.convert_samples();

        // Positions initiales relatives (cf. note v0.9.5++ dans `play_3d`).
        let right = st.listener.right;
        let ear_sep = 0.1;
        let left_ear = [-right.x * ear_sep, -right.y * ear_sep, -right.z * ear_sep];
        let right_ear = [right.x * ear_sep, right.y * ear_sep, right.z * ear_sep];
        let rel = emitter.position - st.listener.position;
        let scale = 0.01_f32;
        let emitter_pos = [rel.x * scale, rel.y * scale, rel.z * scale];

        let sink =
            SpatialSink::try_new(&self.handle, emitter_pos, left_ear, right_ear).ok()?;
        // `repeat_infinite` boucle au niveau du `Source`, pas du sink —
        // le décodage redémarre à 0 sans discontinuité audible.
        sink.append(source.repeat_infinite());
        // Volume initial : atténuation appliquée dès le 1er sample pour
        // éviter un "pop" à volume plein si le joueur spawne loin.
        let dist = emitter.position.distance(st.listener.position);
        let atten = attenuation(dist, emitter.near_dist, emitter.far_dist);
        let vol = (emitter.volume * atten * st.master_volume).clamp(0.0, 1.0);
        sink.set_volume(vol);

        let h = LoopHandle(st.next_loop);
        st.next_loop = st.next_loop.wrapping_add(1).max(1);
        st.loops.insert(h, LoopedEmitter { sink, emitter });
        Some(h)
    }

    /// Stoppe un loop précis spawné par `play_3d_loop`.  No-op si le
    /// handle n'existe pas (loop déjà stoppé ou jamais lancé).
    pub fn stop_loop(&self, handle: LoopHandle) {
        if let Some(l) = self.state.lock().loops.remove(&handle) {
            l.sink.stop();
        }
    }

    /// Stoppe tous les loops — appelé à un `map_load` ou `disconnect`
    /// pour que les speakers de l'ancienne map se taisent.
    pub fn stop_all_loops(&self) {
        let mut st = self.state.lock();
        let drained: Vec<LoopedEmitter> = st.loops.drain().map(|(_, v)| v).collect();
        drop(st);
        for l in drained {
            l.sink.stop();
        }
    }

    /// Nombre de loops 3D actifs — utile pour HUD debug / tests.
    pub fn loop_count(&self) -> usize {
        self.state.lock().loops.len()
    }

    /// Définit (ou remplace) la musique de fond en loop.
    pub fn play_music(&self, bytes: Vec<u8>) -> Result<()> {
        let cursor = Cursor::new(bytes);
        let decoder = Decoder::new(cursor)
            .map_err(|e| Error::Parse(format!("audio: décodage musique échoué: {e}")))?;
        let sink = Sink::try_new(&self.handle)
            .map_err(|e| Error::Renderer(format!("audio: Sink failed: {e}")))?;
        let mut st = self.state.lock();
        sink.set_volume(st.music_volume * st.master_volume);
        sink.append(decoder.repeat_infinite());
        // remplace l'ancienne piste
        if let Some(old) = st.music_sink.take() {
            old.stop();
        }
        st.music_sink = Some(sink);
        Ok(())
    }

    pub fn stop_music(&self) {
        if let Some(s) = self.state.lock().music_sink.take() {
            s.stop();
        }
    }

    pub fn set_music_volume(&self, v: f32) {
        let mut st = self.state.lock();
        st.music_volume = v.clamp(0.0, 1.0);
        let vol = st.music_volume * st.master_volume;
        if let Some(s) = &st.music_sink {
            s.set_volume(vol);
        }
    }

    /// À appeler une fois par frame : purge les canaux qui ont fini et
    /// rafraîchit les positions listener/volume des loops 3D actifs —
    /// c'est ce qui permet au panning et à l'atténuation de suivre
    /// dynamiquement le mouvement du joueur sans avoir à re-démarrer
    /// les sinks.
    pub fn tick(&self) {
        let mut st = self.state.lock();
        for slot in st.channels.iter_mut() {
            if let Some(ch) = slot {
                if ch.sink.empty() {
                    *slot = None;
                }
            }
        }

        // Snapshot des paramètres listener avant la boucle pour éviter un
        // double-borrow sur `st` (on lit `listener` / `master_volume` et
        // on écrit dans `loops`).  Coords relatives (cf. note `play_3d`).
        let pos = st.listener.position;
        let right = st.listener.right;
        let master = st.master_volume;
        let ear_sep = 0.1;
        let left_ear = [-right.x * ear_sep, -right.y * ear_sep, -right.z * ear_sep];
        let right_ear = [right.x * ear_sep, right.y * ear_sep, right.z * ear_sep];
        let scale = 0.01_f32;
        for l in st.loops.values() {
            l.sink.set_left_ear_position(left_ear);
            l.sink.set_right_ear_position(right_ear);
            let rel = l.emitter.position - pos;
            l.sink.set_emitter_position([rel.x * scale, rel.y * scale, rel.z * scale]);
            let dist = l.emitter.position.distance(pos);
            let atten = attenuation(dist, l.emitter.near_dist, l.emitter.far_dist);
            let v = (l.emitter.volume * atten * master).clamp(0.0, 1.0);
            l.sink.set_volume(v);
        }
    }

    /// Canaux actuellement occupés.
    pub fn active_channels(&self) -> usize {
        self.state.lock().channels.iter().filter(|c| c.is_some()).count()
    }

    /// Choisit un canal libre, sinon vole le plus bas prioritaire si la
    /// priorité demandée est strictement supérieure. Renvoie `None` si rien
    /// ne peut être alloué.
    fn pick_channel(channels: &mut [Option<Channel>], wanted: Priority) -> Option<usize> {
        if let Some(i) = channels.iter().position(|c| c.is_none()) {
            return Some(i);
        }
        // Tous occupés : cherche le plus bas prioritaire.
        let (idx, min_prio) = channels
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_ref().map(|ch| (i, ch.priority)))
            .min_by_key(|(_, p)| *p)?;
        if wanted > min_prio {
            if let Some(old) = channels[idx].take() {
                old.sink.stop();
            }
            Some(idx)
        } else {
            None
        }
    }
}

/// Atténuation Q3 — linéaire de `near` à `far`, constante 1 en-dessous.
pub fn attenuation(dist: f32, near: f32, far: f32) -> f32 {
    if dist <= near {
        1.0
    } else if dist >= far {
        0.0
    } else {
        1.0 - (dist - near) / (far - near)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atten_near_is_full() {
        assert!((attenuation(0.0, 64.0, 1024.0) - 1.0).abs() < 1e-6);
        assert!((attenuation(64.0, 64.0, 1024.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn atten_far_is_zero() {
        assert_eq!(attenuation(1024.0, 64.0, 1024.0), 0.0);
        assert_eq!(attenuation(9999.0, 64.0, 1024.0), 0.0);
    }

    #[test]
    fn atten_linear_mid() {
        // 544 = (64 + 1024) / 2
        assert!((attenuation(544.0, 64.0, 1024.0) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn priority_ordering() {
        assert!(Priority::VoiceOver > Priority::Weapon);
        assert!(Priority::Weapon > Priority::Normal);
        assert!(Priority::Normal > Priority::Ambient);
    }
}
