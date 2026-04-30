//! Format **wire** des messages applicatifs entre client et serveur.
//!
//! Encodage : binaire little-endian, manuel (`bytes::Buf` / `BufMut`),
//! pas de `serde` — on veut un layout déterministe, compact, et auditable
//! champ-par-champ. Les flottants `Vec3` passent en `f32` brut (12 octets) ;
//! les angles en `i16` quantifié sur 65536 / 360°.
//!
//! ## Cadrage des messages
//!
//! Chaque payload (post-NetChannel) commence par un octet de **type** :
//!
//! | Tag | Direction        | Contenu                                |
//! |----:|:-----------------|:---------------------------------------|
//! |  1  | client → serveur | [`ClientPacket`] : batch de `UserCmd`  |
//! |  2  | serveur → client | [`Snapshot`]   : état monde courant    |
//!
//! Tous les autres tags sont réservés (chat, vote, demo events…) — un peer
//! recevant un tag inconnu doit logger et drop le paquet, pas crash.
//!
//! ## Pourquoi pas Q3 binary-compatible ?
//!
//! Le netcode Q3 original utilise du bit-packing (`MSG_WriteBits`) avec un
//! delta-compression sur les champs `entityState_t` / `playerState_t`
//! déclaré statiquement. C'est très compact (~30 bytes/snapshot) mais
//! pénible à maintenir. On préfère ici un format byte-aligné, plus large
//! (~50-60 octets/joueur) mais lisible — la delta-compression reste
//! envisageable plus tard via XOR baseline + bitset (cf. roadmap netcode).
//!
//! ## Versionning
//!
//! [`PROTOCOL_VERSION`] est négocié dans le `userinfo` du paquet `connect`
//! OOB. Les versions divergentes refusent la connexion plutôt que d'inter-
//! préter du flux mal cadré — un mismatch silencieux donne des comportements
//! impossible à débugger en prod.

use bytes::{Buf, BufMut, BytesMut};
use q3_common::Error;

/// Numéro de version du protocole applicatif. Incrémenter à toute
/// modification non-rétrocompatible des structures ci-dessous (ajout de
/// champ, changement d'enum, etc.).
pub const PROTOCOL_VERSION: u32 = 1;

/// Tag de paquet client → serveur (batch de `UserCmd`).
pub const TAG_CLIENT_PACKET: u8 = 1;
/// Tag de paquet serveur → client (snapshot **full**).
pub const TAG_SNAPSHOT: u8 = 2;
/// Tag de paquet serveur → client (snapshot **delta**, à appliquer contre
/// une baseline full préalablement reçue par le client).
pub const TAG_SNAPSHOT_DELTA: u8 = 3;

/// Nombre maximal d'`UserCmd` accumulés dans un même paquet client.
/// On en envoie quelques uns à la suite (les non-acquittés depuis le
/// dernier snapshot) pour qu'une perte simple de paquet n'introduise pas
/// de hitch d'input. Q3 plafonne similairement à `MAX_PACKET_USERCMDS`.
pub const MAX_USERCMDS_PER_PACKET: usize = 16;

/// Nombre maximal de joueurs dans un snapshot. Q3 plafonne aussi à 64
/// (`MAX_CLIENTS`) — on tient confortablement sous le MTU même en faisant
/// 64 × `PlayerState` (~50 octets) = 3.2 KiB, donc `NetChannel` fragmente.
pub const MAX_PLAYERS_PER_SNAPSHOT: usize = 64;

/// Nombre maximal d'entités dynamiques (projectiles principalement) dans
/// un snapshot. Au-delà, on tronque — un duel typique a < 10 projectiles
/// en l'air, 256 est large.
pub const MAX_ENTITIES_PER_SNAPSHOT: usize = 256;

// ---------------------------------------------------------------------------
// Boutons du joueur (bitset)
// ---------------------------------------------------------------------------

/// Bits de `UserCmd::buttons`. Compact pour ne pas saturer la bande
/// passante avec 6 booléens. Un seul `u16` laisse de la marge pour les
/// scripts d'input futurs (taunts, scores, gestures…).
pub mod buttons {
    pub const FIRE: u16 = 1 << 0;
    pub const JUMP: u16 = 1 << 1;
    pub const CROUCH: u16 = 1 << 2;
    pub const WALK: u16 = 1 << 3;
    pub const USE_HOLDABLE: u16 = 1 << 4;
}

// ---------------------------------------------------------------------------
// UserCmd
// ---------------------------------------------------------------------------

/// Une commande d'input client. Représente l'état des touches/souris pour
/// **un** tick de physique côté client (8 ms à 125 Hz). Le client en
/// accumule plusieurs entre deux paquets et les rejoue dans l'ordre côté
/// serveur — c'est ce qui permet une simulation déterministe dans les
/// deux sens (prédiction client, autorité serveur).
///
/// ## Quantification
/// * Axes `forward`/`side`/`up` : `i8` ∈ [-127, 127]. Côté client on les
///   produit avec `(axis * 127.0).clamp(-127, 127) as i8`.
/// * Angles : `i16` quantifié sur 65 536 / 360° = ~0.0055°/pas. Très
///   au-delà de la précision perceptible humaine (~0.1°), donc lossless
///   en pratique. Pour décoder : `i as f32 * 360.0 / 65536.0`.
///
/// La taille fil = 4 + 3 + 2 + 6 + 1 + 1 = **17 octets** par cmd.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UserCmd {
    /// Numéro logique de la commande, monotone côté client. Sert
    /// d'**ack** pour la prédiction (dernier `cmd_number` que le serveur
    /// a appliqué, renvoyé dans `Snapshot::ack_cmd`).
    pub cmd_number: u32,
    pub forward: i8,
    pub side: i8,
    pub up: i8,
    pub buttons: u16,
    /// Pitch / yaw / roll en `i16` quantifiés.
    pub view_pitch: i16,
    pub view_yaw: i16,
    pub view_roll: i16,
    /// Durée de simulation représentée par cette commande, en
    /// millisecondes. À 125 Hz = 8 ms. Cap à `u8::MAX` (255 ms) — au-delà
    /// le client a hitché et on coupe net pour ne pas téléporter.
    pub delta_ms: u8,
    /// Arme active pendant ce tick (slot 0..9).
    pub weapon: u8,
}

impl UserCmd {
    /// Taille d'une `UserCmd` sur le fil, en octets.
    pub const WIRE_SIZE: usize = 4 + 3 + 2 + 6 + 1 + 1;

    /// Encode `i16` quantifié à partir d'un angle en degrés.
    #[inline]
    pub fn quantize_angle(deg: f32) -> i16 {
        // Réduit modulo 360 pour éviter les overflows à grande dérive
        // (le yaw d'un joueur qui tourne longtemps peut s'accumuler).
        let wrapped = deg.rem_euclid(360.0);
        let q = (wrapped / 360.0 * 65536.0).round() as i32;
        // 65536 → 0 après wrap. Cast explicite via i32→i16 pour wrapping
        // safe sur les valeurs limites.
        (q as i32 & 0xFFFF) as i16
    }

    /// Décode l'angle en degrés depuis `i16` quantifié, dans [0, 360).
    #[inline]
    pub fn dequantize_angle(q: i16) -> f32 {
        // Cast en u16 pour traiter comme valeur 16-bit non-signée — sinon
        // les angles > 180° (encodés en complément à 2) ressortiraient
        // négatifs et casseraient la simulation côté client.
        (q as u16 as f32) * 360.0 / 65536.0
    }

    /// Encode l'axe i8 depuis un f32 dans [-1, 1].
    #[inline]
    pub fn quantize_axis(v: f32) -> i8 {
        (v.clamp(-1.0, 1.0) * 127.0).round() as i8
    }

    /// Décode l'axe vers f32 dans [-1, 1].
    #[inline]
    pub fn dequantize_axis(v: i8) -> f32 {
        v as f32 / 127.0
    }

    fn write(&self, buf: &mut BytesMut) {
        buf.put_u32_le(self.cmd_number);
        buf.put_i8(self.forward);
        buf.put_i8(self.side);
        buf.put_i8(self.up);
        buf.put_u16_le(self.buttons);
        buf.put_i16_le(self.view_pitch);
        buf.put_i16_le(self.view_yaw);
        buf.put_i16_le(self.view_roll);
        buf.put_u8(self.delta_ms);
        buf.put_u8(self.weapon);
    }

    fn read(cur: &mut &[u8]) -> Result<Self, Error> {
        if cur.remaining() < Self::WIRE_SIZE {
            return Err(Error::Network("UserCmd: tronqué".into()));
        }
        let cmd_number = cur.get_u32_le();
        let forward = cur.get_i8();
        let side = cur.get_i8();
        let up = cur.get_i8();
        let buttons = cur.get_u16_le();
        let view_pitch = cur.get_i16_le();
        let view_yaw = cur.get_i16_le();
        let view_roll = cur.get_i16_le();
        let delta_ms = cur.get_u8();
        let weapon = cur.get_u8();
        Ok(Self {
            cmd_number,
            forward,
            side,
            up,
            buttons,
            view_pitch,
            view_yaw,
            view_roll,
            delta_ms,
            weapon,
        })
    }
}

// ---------------------------------------------------------------------------
// ClientPacket — batch d'UserCmd
// ---------------------------------------------------------------------------

/// Paquet applicatif client → serveur, encapsulé dans un `NetChannel`.
///
/// Format :
/// ```text
/// u8   tag = TAG_CLIENT_PACKET
/// u32  server_time_ack    (ms — dernier server_time vu, 0 si jamais)
/// u8   count               (nb d'UserCmd suivantes)
/// UserCmd[count]
/// ```
///
/// Le batching n'est pas du « replay » : le serveur applique chaque
/// commande dans l'ordre de `cmd_number`, et celles déjà appliquées
/// (cf. `last_cmd_applied`) sont ignorées. C'est ce qui permet de gérer
/// la perte de paquets : tant qu'au moins un paquet sur N arrive, aucune
/// commande n'est perdue.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClientPacket {
    /// Dernier `server_time` reçu du serveur (echo, sert au RTT et à la
    /// résolution de prédiction). 0 = client n'a encore rien reçu.
    pub server_time_ack: u32,
    pub cmds: Vec<UserCmd>,
}

impl ClientPacket {
    pub fn encode(&self) -> Result<Vec<u8>, Error> {
        if self.cmds.len() > MAX_USERCMDS_PER_PACKET {
            return Err(Error::Network(format!(
                "ClientPacket: trop d'UserCmd ({} > {})",
                self.cmds.len(),
                MAX_USERCMDS_PER_PACKET
            )));
        }
        let mut buf =
            BytesMut::with_capacity(1 + 4 + 1 + self.cmds.len() * UserCmd::WIRE_SIZE);
        buf.put_u8(TAG_CLIENT_PACKET);
        buf.put_u32_le(self.server_time_ack);
        buf.put_u8(self.cmds.len() as u8);
        for c in &self.cmds {
            c.write(&mut buf);
        }
        Ok(buf.to_vec())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let mut cur = bytes;
        if cur.remaining() < 1 + 4 + 1 {
            return Err(Error::Network("ClientPacket: header tronqué".into()));
        }
        let tag = cur.get_u8();
        if tag != TAG_CLIENT_PACKET {
            return Err(Error::Network(format!(
                "ClientPacket: tag {tag} != {TAG_CLIENT_PACKET}"
            )));
        }
        let server_time_ack = cur.get_u32_le();
        let count = cur.get_u8() as usize;
        if count > MAX_USERCMDS_PER_PACKET {
            return Err(Error::Network(format!(
                "ClientPacket: count {count} > max"
            )));
        }
        let mut cmds = Vec::with_capacity(count);
        for _ in 0..count {
            cmds.push(UserCmd::read(&mut cur)?);
        }
        Ok(Self {
            server_time_ack,
            cmds,
        })
    }
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// État sérialisable d'un joueur (humain ou bot) — 76 octets sur le fil.
///
/// Diffère du `PlayerMove` interne : on sérialise uniquement ce que les
/// **autres clients** doivent voir / ce que le client local a besoin pour
/// reconstruire son HUD. Pas de champs de prédiction interne (`bob_phase`,
/// timers viewmodel…) — chaque client fait son propre rendu de ces aspects
/// purement visuels à partir de l'origine + vélocité.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PlayerState {
    /// Slot serveur du joueur (0..MAX_CLIENTS-1). Stable pour la durée
    /// de la connexion. Le client compare avec son propre `client_slot`
    /// dans `Snapshot` pour savoir quel `PlayerState` est « lui ».
    pub slot: u8,
    /// Bitset `PLAYER_FLAG_*`.
    pub flags: u8,
    pub health: i16,
    pub armor: i16,
    pub weapon: u8,
    /// Bitset `POWERUP_*` — Quad / Haste / Regen / Invis / BattleSuit / Flight.
    pub powerups: u8,
    pub frags: i16,
    pub deaths: i16,
    pub origin: [f32; 3],
    pub velocity: [f32; 3],
    /// Pitch / yaw / roll, degrés. Pas quantifiés ici (on est côté snapshot
    /// 20 Hz, le coût d'envoyer 12 octets de plus est négligeable et ça
    /// économise un round-trip d'erreur de quantification visible sur les
    /// rotations rapides des autres joueurs).
    pub view_angles: [f32; 3],
    /// Stock de munitions par slot d'arme (0..9). 20 octets sur le fil.
    /// Inclus pour TOUS les joueurs même si seul le local a vraiment
    /// besoin du détail — coût négligeable (160 octets pour 8 joueurs)
    /// et la delta-compression élimine les duplicats tick-à-tick.
    pub ammo: [i16; 10],
    /// Équipe en mode TDM/CTF : `0` = free / FFA (pas d'équipe),
    /// `1` = red, `2` = blue. Détermine la couleur du tint MD3 côté
    /// rendu et les règles de friendly fire (TODO). En FFA, tous les
    /// slots ont team=0 et sont rendus avec leur tint slot-id habituel.
    pub team: u8,
}

/// Constantes équipe — utilisables côté server et client sans dépendance
/// au format wire.
pub mod team {
    pub const FREE: u8 = 0;
    pub const RED: u8 = 1;
    pub const BLUE: u8 = 2;
}

/// Bits de `PlayerState::powerups`. Aligné sur l'ordre `PowerupKind`
/// du client pour faciliter la conversion (Quad en bit 0 = mêmes
/// indices). Cap à 8 powerups simultanés (plus que les 6 actuels) →
/// `u8` suffit.
pub mod powerup_flags {
    pub const QUAD_DAMAGE: u8 = 1 << 0;
    pub const HASTE: u8 = 1 << 1;
    pub const REGENERATION: u8 = 1 << 2;
    pub const BATTLE_SUIT: u8 = 1 << 3;
    pub const INVISIBILITY: u8 = 1 << 4;
    pub const FLIGHT: u8 = 1 << 5;
}

/// Bits de `PlayerState::flags`.
pub mod player_flags {
    pub const ON_GROUND: u8 = 1 << 0;
    pub const CROUCHING: u8 = 1 << 1;
    pub const DEAD: u8 = 1 << 2;
    /// Le joueur est un bot piloté par le serveur (le client ne tente pas
    /// de prédire son input). Aide au rendu : on n'affiche pas son nom
    /// avec « (bot) » mais on peut adapter le label HUD.
    pub const BOT: u8 = 1 << 3;
    /// Le joueur a tiré dans la fenêtre récente (typiquement 250 ms).
    /// Côté client, sert à déclencher l'anim `TORSO_ATTACK` du remote
    /// MD3 — sans ça les autres joueurs ne montrent jamais de geste de
    /// tir, ce qui rend les duels moins lisibles.
    pub const RECENTLY_FIRED: u8 = 1 << 4;
    /// Le joueur est spectateur — connecté mais ne participe pas au
    /// combat. Le serveur ignore ses dégâts entrants/sortants et le
    /// passage sur les pickups. Le client masque son HUD ammo/weapon
    /// et peut afficher un overlay « SPECTATING ».
    pub const SPECTATOR: u8 = 1 << 5;
}

impl PlayerState {
    pub const WIRE_SIZE: usize = 1 + 1 + 2 + 2 + 1 + 1 + 2 + 2 + 12 + 12 + 12 + 20 + 1;

    fn write(&self, buf: &mut BytesMut) {
        buf.put_u8(self.slot);
        buf.put_u8(self.flags);
        buf.put_i16_le(self.health);
        buf.put_i16_le(self.armor);
        buf.put_u8(self.weapon);
        buf.put_u8(self.powerups);
        buf.put_i16_le(self.frags);
        buf.put_i16_le(self.deaths);
        for v in &self.origin {
            buf.put_f32_le(*v);
        }
        for v in &self.velocity {
            buf.put_f32_le(*v);
        }
        for v in &self.view_angles {
            buf.put_f32_le(*v);
        }
        for v in &self.ammo {
            buf.put_i16_le(*v);
        }
        buf.put_u8(self.team);
    }

    fn read(cur: &mut &[u8]) -> Result<Self, Error> {
        if cur.remaining() < Self::WIRE_SIZE {
            return Err(Error::Network("PlayerState: tronqué".into()));
        }
        let slot = cur.get_u8();
        let flags = cur.get_u8();
        let health = cur.get_i16_le();
        let armor = cur.get_i16_le();
        let weapon = cur.get_u8();
        let powerups = cur.get_u8();
        let frags = cur.get_i16_le();
        let deaths = cur.get_i16_le();
        let origin = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
        let velocity = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
        let view_angles = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
        let mut ammo = [0i16; 10];
        for v in &mut ammo {
            *v = cur.get_i16_le();
        }
        let team = cur.get_u8();
        Ok(Self {
            slot,
            flags,
            health,
            armor,
            weapon,
            powerups,
            frags,
            deaths,
            origin,
            velocity,
            view_angles,
            ammo,
            team,
        })
    }
}

/// Type d'entité dynamique sérialisée. Volontairement compact : on
/// n'envoie pas les pickups statiques (le client a la map et son layout),
/// uniquement ce qui se déplace ou apparaît/disparaît dynamiquement.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKindWire {
    Rocket = 1,
    Plasma = 2,
    Grenade = 3,
    /// Une particule d'explosion / smoke-puff visible — utile pour les
    /// gros effets dont l'origine doit rester synchronisée
    /// (rocket impact). Évènements ponctuels passent par un autre canal
    /// dans une future version ; pour v1 on les rend persistants courts.
    Explosion = 4,
    /// BFG projectile — gros, lent, splash énorme (300dmg/200rad).
    /// Visuel typique : sphère verte avec halo.
    Bfg = 5,
}

impl EntityKindWire {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            1 => Self::Rocket,
            2 => Self::Plasma,
            3 => Self::Grenade,
            4 => Self::Explosion,
            5 => Self::Bfg,
            _ => return None,
        })
    }
}

/// État d'une entité dynamique (projectile, explosion). 26 octets fil.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntityState {
    /// ID stable serveur — incrémenté à chaque spawn d'entité dynamique.
    /// L'absence d'un ID dans le snapshot suivant signifie que l'entité
    /// a été détruite : le client peut faire une dernière interpolation
    /// + jouer un effet de fin (explosion, fade).
    pub id: u32,
    pub kind: EntityKindWire,
    /// Owner (slot joueur) — utilisé pour le kill-feed. 255 = pas d'owner
    /// (entité environnementale, e.g. lave projetée — non utilisé en v1).
    pub owner: u8,
    pub origin: [f32; 3],
    pub velocity: [f32; 3],
}

impl EntityState {
    pub const WIRE_SIZE: usize = 4 + 1 + 1 + 12 + 12;

    fn write(&self, buf: &mut BytesMut) {
        buf.put_u32_le(self.id);
        buf.put_u8(self.kind as u8);
        buf.put_u8(self.owner);
        for v in &self.origin {
            buf.put_f32_le(*v);
        }
        for v in &self.velocity {
            buf.put_f32_le(*v);
        }
    }

    fn read(cur: &mut &[u8]) -> Result<Self, Error> {
        if cur.remaining() < Self::WIRE_SIZE {
            return Err(Error::Network("EntityState: tronqué".into()));
        }
        let id = cur.get_u32_le();
        let kind_raw = cur.get_u8();
        let kind = EntityKindWire::from_u8(kind_raw)
            .ok_or_else(|| Error::Network(format!("EntityState: kind {kind_raw} inconnu")))?;
        let owner = cur.get_u8();
        let origin = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
        let velocity = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
        Ok(Self {
            id,
            kind,
            owner,
            origin,
            velocity,
        })
    }
}

/// Snapshot complet — état monde envoyé du serveur à un client.
///
/// Format :
/// ```text
/// u8   tag = TAG_SNAPSHOT
/// u32  server_time   (ms écoulées depuis le boot du serveur)
/// u32  ack_cmd       (dernier UserCmd appliqué pour CE client — 0 si neuf)
/// u8   client_slot   (slot du destinataire dans players[])
/// u16  player_count
/// PlayerState[player_count]
/// u16  entity_count
/// EntityState[entity_count]
/// u16  pickup_count
/// PickupState[pickup_count]
/// ```
///
/// `pickup_count` peut être 0 si rien n'a changé — pour v1 on l'envoie
/// systématiquement (full snapshot). En v2 on n'enverra que le delta.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Snapshot {
    pub server_time: u32,
    pub ack_cmd: u32,
    pub client_slot: u8,
    pub players: Vec<PlayerState>,
    pub entities: Vec<EntityState>,
    pub pickups: Vec<PickupState>,
    /// Évènements ponctuels survenus depuis le précédent snapshot
    /// (explosions, kills…). Best-effort — un client qui rate ce
    /// snapshot rate aussi ses évènements. Acceptable pour les effets
    /// cosmétiques.
    pub events: Vec<ServerEvent>,
    /// Métadonnées des slots (noms). Inclus dans tous les snapshots
    /// full ; les deltas héritent de leur baseline. Coût ~17 octets/slot.
    pub players_info: Vec<PlayerInfo>,
}

/// Métadonnées d'un joueur — nom essentiellement. Diffusé uniquement
/// dans les snapshots **full** (toutes les ~1 s). Les deltas héritent
/// du `players_info` de leur baseline. Conséquence : un nouveau joueur
/// peut prendre jusqu'à 1 s pour apparaître nominativement dans le
/// kill-feed des autres ; acceptable pour v1.
///
/// Le nom est stocké en `[u8; 16]` UTF-8 (pas plus de 15 octets utiles
/// + nul terminateur). Plus court → padding zéro. Plus long → tronqué.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerInfo {
    pub slot: u8,
    pub name: [u8; 16],
}

impl PlayerInfo {
    pub const WIRE_SIZE: usize = 1 + 16;

    /// Construit un `PlayerInfo` depuis un nom Rust ; tronque à 16
    /// octets ASCII/UTF-8 et pad de zéros.
    pub fn new(slot: u8, name: &str) -> Self {
        let mut buf = [0u8; 16];
        let bytes = name.as_bytes();
        let n = bytes.len().min(16);
        buf[..n].copy_from_slice(&bytes[..n]);
        Self { slot, name: buf }
    }

    /// Renvoie le nom sous forme de String, en stripant les zéros de
    /// padding et en remplaçant les bytes invalides UTF-8 par un
    /// caractère de remplacement (U+FFFD).
    pub fn name_string(&self) -> String {
        let end = self
            .name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.name.len());
        String::from_utf8_lossy(&self.name[..end]).into_owned()
    }

    fn write(&self, buf: &mut BytesMut) {
        buf.put_u8(self.slot);
        buf.put_slice(&self.name);
    }

    fn read(cur: &mut &[u8]) -> Result<Self, Error> {
        if cur.remaining() < Self::WIRE_SIZE {
            return Err(Error::Network("PlayerInfo: tronqué".into()));
        }
        let slot = cur.get_u8();
        let mut name = [0u8; 16];
        cur.copy_to_slice(&mut name);
        Ok(Self { slot, name })
    }
}

/// État d'un pickup — uniquement diffusé quand son état change pour
/// économiser de la bande passante. ID = index dans la liste des pickups
/// du serveur (stable car la liste est construite au chargement de map).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PickupState {
    pub id: u16,
    /// 1 = disponible, 0 = ramassé en attente de respawn.
    pub available: u8,
}

impl PickupState {
    pub const WIRE_SIZE: usize = 2 + 1;

    fn write(&self, buf: &mut BytesMut) {
        buf.put_u16_le(self.id);
        buf.put_u8(self.available);
    }

    fn read(cur: &mut &[u8]) -> Result<Self, Error> {
        if cur.remaining() < Self::WIRE_SIZE {
            return Err(Error::Network("PickupState: tronqué".into()));
        }
        let id = cur.get_u16_le();
        let available = cur.get_u8();
        Ok(Self { id, available })
    }
}

impl Snapshot {
    pub fn encode(&self) -> Result<Vec<u8>, Error> {
        if self.players.len() > MAX_PLAYERS_PER_SNAPSHOT {
            return Err(Error::Network(format!(
                "Snapshot: trop de joueurs ({} > {})",
                self.players.len(),
                MAX_PLAYERS_PER_SNAPSHOT
            )));
        }
        if self.entities.len() > MAX_ENTITIES_PER_SNAPSHOT {
            return Err(Error::Network(format!(
                "Snapshot: trop d'entités ({} > {})",
                self.entities.len(),
                MAX_ENTITIES_PER_SNAPSHOT
            )));
        }
        let cap = 1
            + 4
            + 4
            + 1
            + 2
            + self.players.len() * PlayerState::WIRE_SIZE
            + 2
            + self.entities.len() * EntityState::WIRE_SIZE
            + 2
            + self.pickups.len() * PickupState::WIRE_SIZE
            + 1
            + self.events.len() * 14; // ~max event size
        let mut buf = BytesMut::with_capacity(cap);
        buf.put_u8(TAG_SNAPSHOT);
        buf.put_u32_le(self.server_time);
        buf.put_u32_le(self.ack_cmd);
        buf.put_u8(self.client_slot);
        buf.put_u16_le(self.players.len() as u16);
        for p in &self.players {
            p.write(&mut buf);
        }
        buf.put_u16_le(self.entities.len() as u16);
        for e in &self.entities {
            e.write(&mut buf);
        }
        buf.put_u16_le(self.pickups.len() as u16);
        for p in &self.pickups {
            p.write(&mut buf);
        }
        write_events(&mut buf, &self.events)?;
        // players_info en queue (toujours présent, même si vide).
        // Codé en u8 count car MAX_PLAYERS = 64.
        if self.players_info.len() > MAX_PLAYERS_PER_SNAPSHOT {
            return Err(Error::Network("Snapshot: trop de players_info".into()));
        }
        buf.put_u8(self.players_info.len() as u8);
        for pi in &self.players_info {
            pi.write(&mut buf);
        }
        Ok(buf.to_vec())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let mut cur = bytes;
        if cur.remaining() < 1 + 4 + 4 + 1 + 2 {
            return Err(Error::Network("Snapshot: header tronqué".into()));
        }
        let tag = cur.get_u8();
        if tag != TAG_SNAPSHOT {
            return Err(Error::Network(format!(
                "Snapshot: tag {tag} != {TAG_SNAPSHOT}"
            )));
        }
        let server_time = cur.get_u32_le();
        let ack_cmd = cur.get_u32_le();
        let client_slot = cur.get_u8();
        let player_count = cur.get_u16_le() as usize;
        if player_count > MAX_PLAYERS_PER_SNAPSHOT {
            return Err(Error::Network(format!(
                "Snapshot: player_count {player_count} > max"
            )));
        }
        let mut players = Vec::with_capacity(player_count);
        for _ in 0..player_count {
            players.push(PlayerState::read(&mut cur)?);
        }
        if cur.remaining() < 2 {
            return Err(Error::Network("Snapshot: entity_count tronqué".into()));
        }
        let entity_count = cur.get_u16_le() as usize;
        if entity_count > MAX_ENTITIES_PER_SNAPSHOT {
            return Err(Error::Network(format!(
                "Snapshot: entity_count {entity_count} > max"
            )));
        }
        let mut entities = Vec::with_capacity(entity_count);
        for _ in 0..entity_count {
            entities.push(EntityState::read(&mut cur)?);
        }
        if cur.remaining() < 2 {
            return Err(Error::Network("Snapshot: pickup_count tronqué".into()));
        }
        let pickup_count = cur.get_u16_le() as usize;
        let mut pickups = Vec::with_capacity(pickup_count);
        for _ in 0..pickup_count {
            pickups.push(PickupState::read(&mut cur)?);
        }
        let events = read_events(&mut cur)?;
        // players_info — peut être absent dans des snapshots produits
        // par une version ancienne du protocole : on tolère le tronquage
        // ici pour préserver l'interopérabilité ascendante. En production
        // un client à jour reçoit toujours ce champ.
        let players_info = if cur.remaining() >= 1 {
            let n = cur.get_u8() as usize;
            if n > MAX_PLAYERS_PER_SNAPSHOT {
                return Err(Error::Network(format!(
                    "Snapshot: players_info count {n} > max"
                )));
            }
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(PlayerInfo::read(&mut cur)?);
            }
            out
        } else {
            Vec::new()
        };
        Ok(Self {
            server_time,
            ack_cmd,
            client_slot,
            players,
            entities,
            pickups,
            events,
            players_info,
        })
    }
}

// ---------------------------------------------------------------------------
// ServerEvent — évènements ponctuels (explosion, kill, hit confirm)
// ---------------------------------------------------------------------------

/// Évènement transmis dans `Snapshot::events` / `SnapshotDelta::events`.
/// Best-effort : si le client manque le snapshot, il rate l'évènement —
/// acceptable pour les effets cosmétiques (explosion visuelle, sons).
/// Pour les états persistants (santé, frags) on utilise `PlayerState`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ServerEvent {
    /// Une explosion vient de se produire à `pos`. `kind` distingue le
    /// type pour le client (couleur dlight, son joué, sparks colorés).
    Explosion { pos: [f32; 3], kind: ExplosionKind },
    /// Un joueur est mort. Sert au kill-feed.
    PlayerKilled {
        victim: u8,
        killer: u8,
        weapon: u8,
    },
    /// Trail visuel d'un tir hitscan instantané (railgun typiquement).
    /// `from` = point d'émission (œil + petit offset muzzle), `to` =
    /// point d'impact (joueur ou mur). `owner` permet au client de
    /// choisir une couleur perso si la personnalisation team / skin est
    /// implémentée plus tard ; en v1 tous les rails sont bleu cyan.
    RailTrail {
        from: [f32; 3],
        to: [f32; 3],
        owner: u8,
    },
    /// Le match vient de se terminer (fraglimit ou timelimit atteint).
    /// `winner` = slot du gagnant, ou 255 pour égalité. Le client passe
    /// en mode intermission (caméra figée, scoreboard plein écran).
    MatchEnded { winner: u8 },
    /// Un nouveau match commence — clear le kill-feed, reset HUD.
    MatchStarted,
    /// Beam de lightning gun — segment droit zigzag visible pendant
    /// quelques frames côté client. Émis à chaque tick FIRE quand le
    /// joueur tient l'arme (~50/s en cadence Q3 — wire-bandwidth-OK
    /// car 25 octets × 50 = 1.25 KiB/s pour 1 tireur).
    LightningBeam {
        from: [f32; 3],
        to: [f32; 3],
        owner: u8,
    },
    /// Évènement sonore positionnel — pickup chime, jump pad whoosh,
    /// teleport, etc. `id` est un `SoundId::*` u8. `pos` = lieu d'émission
    /// pour le mixage 3D côté client. 14 octets sur le fil.
    Sound { id: u8, pos: [f32; 3] },
    /// Ligne de chat globale émise par un joueur. Le serveur reçoit
    /// la requête via OOB `say "<message>"`, la diffuse ensuite à tous
    /// les clients dans le snapshot suivant. Le `slot` identifie
    /// l'auteur (le client résout le nom via sa table `remote_names`).
    /// Message tronqué/padded à 96 octets, UTF-8 lossy à la décode.
    Chat { slot: u8, message: [u8; 96] },
}

impl ServerEvent {
    /// Construit un `ServerEvent::Chat` depuis un nom Rust ; tronque à
    /// 96 octets et pad de zéros.
    pub fn new_chat(slot: u8, message: &str) -> Self {
        let mut buf = [0u8; 96];
        let bytes = message.as_bytes();
        let n = bytes.len().min(96);
        buf[..n].copy_from_slice(&bytes[..n]);
        Self::Chat { slot, message: buf }
    }

    /// Renvoie le message d'un `Chat` event sous forme de String.
    /// Utile côté client pour passer à `chat_feed`.
    pub fn chat_message(&self) -> Option<(u8, String)> {
        if let Self::Chat { slot, message } = self {
            let end = message.iter().position(|&b| b == 0).unwrap_or(96);
            Some((*slot, String::from_utf8_lossy(&message[..end]).into_owned()))
        } else {
            None
        }
    }
}

/// IDs stables pour `ServerEvent::Sound`. Tout ID inconnu côté client
/// est silencieusement ignoré (logué en debug). Numérotation gappée
/// pour permettre des ajouts par catégorie sans rupture wire.
pub mod sound_id {
    pub const PICKUP_HEALTH: u8 = 1;
    pub const PICKUP_ARMOR: u8 = 2;
    pub const PICKUP_AMMO: u8 = 3;
    pub const PICKUP_POWERUP: u8 = 4;
    pub const PICKUP_WEAPON: u8 = 5;
}

/// Type d'explosion — pour piloter visuels/audio côté client.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplosionKind {
    Rocket = 1,
    Plasma = 2,
    Grenade = 3,
    Bfg = 4,
}

impl ExplosionKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            1 => Self::Rocket,
            2 => Self::Plasma,
            3 => Self::Grenade,
            4 => Self::Bfg,
            _ => return None,
        })
    }
}

const EVENT_TAG_EXPLOSION: u8 = 1;
const EVENT_TAG_PLAYER_KILLED: u8 = 2;
const EVENT_TAG_RAIL_TRAIL: u8 = 3;
const EVENT_TAG_MATCH_ENDED: u8 = 4;
const EVENT_TAG_MATCH_STARTED: u8 = 5;
const EVENT_TAG_LIGHTNING_BEAM: u8 = 6;
const EVENT_TAG_SOUND: u8 = 7;
const EVENT_TAG_CHAT: u8 = 8;

impl ServerEvent {
    fn write(&self, buf: &mut BytesMut) {
        match self {
            Self::Explosion { pos, kind } => {
                buf.put_u8(EVENT_TAG_EXPLOSION);
                for v in pos {
                    buf.put_f32_le(*v);
                }
                buf.put_u8(*kind as u8);
            }
            Self::PlayerKilled {
                victim,
                killer,
                weapon,
            } => {
                buf.put_u8(EVENT_TAG_PLAYER_KILLED);
                buf.put_u8(*victim);
                buf.put_u8(*killer);
                buf.put_u8(*weapon);
            }
            Self::RailTrail { from, to, owner } => {
                buf.put_u8(EVENT_TAG_RAIL_TRAIL);
                for v in from {
                    buf.put_f32_le(*v);
                }
                for v in to {
                    buf.put_f32_le(*v);
                }
                buf.put_u8(*owner);
            }
            Self::MatchEnded { winner } => {
                buf.put_u8(EVENT_TAG_MATCH_ENDED);
                buf.put_u8(*winner);
            }
            Self::MatchStarted => {
                buf.put_u8(EVENT_TAG_MATCH_STARTED);
            }
            Self::LightningBeam { from, to, owner } => {
                buf.put_u8(EVENT_TAG_LIGHTNING_BEAM);
                for v in from {
                    buf.put_f32_le(*v);
                }
                for v in to {
                    buf.put_f32_le(*v);
                }
                buf.put_u8(*owner);
            }
            Self::Sound { id, pos } => {
                buf.put_u8(EVENT_TAG_SOUND);
                buf.put_u8(*id);
                for v in pos {
                    buf.put_f32_le(*v);
                }
            }
            Self::Chat { slot, message } => {
                buf.put_u8(EVENT_TAG_CHAT);
                buf.put_u8(*slot);
                buf.put_slice(message);
            }
        }
    }

    fn read(cur: &mut &[u8]) -> Result<Self, Error> {
        if cur.remaining() < 1 {
            return Err(Error::Network("ServerEvent: tag manquant".into()));
        }
        let tag = cur.get_u8();
        match tag {
            EVENT_TAG_EXPLOSION => {
                if cur.remaining() < 13 {
                    return Err(Error::Network("Explosion: payload tronqué".into()));
                }
                let pos = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
                let kind_raw = cur.get_u8();
                let kind = ExplosionKind::from_u8(kind_raw).ok_or_else(|| {
                    Error::Network(format!("ExplosionKind {kind_raw} inconnu"))
                })?;
                Ok(Self::Explosion { pos, kind })
            }
            EVENT_TAG_PLAYER_KILLED => {
                if cur.remaining() < 3 {
                    return Err(Error::Network("PlayerKilled: payload tronqué".into()));
                }
                let victim = cur.get_u8();
                let killer = cur.get_u8();
                let weapon = cur.get_u8();
                Ok(Self::PlayerKilled {
                    victim,
                    killer,
                    weapon,
                })
            }
            EVENT_TAG_RAIL_TRAIL => {
                if cur.remaining() < 25 {
                    return Err(Error::Network("RailTrail: payload tronqué".into()));
                }
                let from = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
                let to = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
                let owner = cur.get_u8();
                Ok(Self::RailTrail { from, to, owner })
            }
            EVENT_TAG_MATCH_ENDED => {
                if cur.remaining() < 1 {
                    return Err(Error::Network("MatchEnded: payload tronqué".into()));
                }
                let winner = cur.get_u8();
                Ok(Self::MatchEnded { winner })
            }
            EVENT_TAG_MATCH_STARTED => Ok(Self::MatchStarted),
            EVENT_TAG_LIGHTNING_BEAM => {
                if cur.remaining() < 25 {
                    return Err(Error::Network("LightningBeam: payload tronqué".into()));
                }
                let from = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
                let to = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
                let owner = cur.get_u8();
                Ok(Self::LightningBeam { from, to, owner })
            }
            EVENT_TAG_SOUND => {
                if cur.remaining() < 13 {
                    return Err(Error::Network("Sound: payload tronqué".into()));
                }
                let id = cur.get_u8();
                let pos = [cur.get_f32_le(), cur.get_f32_le(), cur.get_f32_le()];
                Ok(Self::Sound { id, pos })
            }
            EVENT_TAG_CHAT => {
                if cur.remaining() < 1 + 96 {
                    return Err(Error::Network("Chat: payload tronqué".into()));
                }
                let slot = cur.get_u8();
                let mut message = [0u8; 96];
                cur.copy_to_slice(&mut message);
                Ok(Self::Chat { slot, message })
            }
            other => Err(Error::Network(format!("ServerEvent tag {other} inconnu"))),
        }
    }
}

/// Cap d'évènements par snapshot — au-delà on tronque (avec warning
/// côté serveur). En pratique on tire 1 explosion par tick max, donc
/// 64 events couvre largement 3 secondes de gros combat.
pub const MAX_EVENTS_PER_SNAPSHOT: usize = 64;

fn write_events(buf: &mut BytesMut, events: &[ServerEvent]) -> Result<(), Error> {
    if events.len() > MAX_EVENTS_PER_SNAPSHOT {
        return Err(Error::Network(format!(
            "events: trop ({} > {})",
            events.len(),
            MAX_EVENTS_PER_SNAPSHOT
        )));
    }
    buf.put_u8(events.len() as u8);
    for e in events {
        e.write(buf);
    }
    Ok(())
}

fn read_events(cur: &mut &[u8]) -> Result<Vec<ServerEvent>, Error> {
    if cur.remaining() < 1 {
        return Err(Error::Network("events_count tronqué".into()));
    }
    let n = cur.get_u8() as usize;
    if n > MAX_EVENTS_PER_SNAPSHOT {
        return Err(Error::Network(format!(
            "events_count {n} > {MAX_EVENTS_PER_SNAPSHOT}"
        )));
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(ServerEvent::read(cur)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// SnapshotDelta — encodage compact contre une baseline
// ---------------------------------------------------------------------------

/// Snapshot delta — variante compacte de [`Snapshot`] qui n'embarque que
/// les `PlayerState` dont au moins un champ a changé depuis la baseline.
///
/// Économie typique : sur 8 joueurs dont 2 bougent et 6 sont idle,
/// `dirty_players` contient 2 entrées au lieu de 8 → ~75 % de bande
/// passante en moins par snapshot.
///
/// # Limitations v1
///
/// - **Pas de delta sur les entités** : la liste est toujours full. Les
///   projectiles spawnent / dieent au tick, leur compression delta a peu
///   de gain.
/// - **Pas de delta sur les pickups** : on les envoie déjà uniquement
///   quand leur état change (cf. broadcast côté serveur), donc la liste
///   est typiquement vide. Pas besoin d'un mécanisme de plus.
/// - **Slots disparus** : si un joueur quitte, son slot reste « zombie »
///   dans la baseline jusqu'au prochain snapshot full (1 s plus tard
///   max). Acceptable — Q3 historique a la même fenêtre.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SnapshotDelta {
    pub server_time: u32,
    pub ack_cmd: u32,
    pub client_slot: u8,
    /// `server_time` du snapshot full de référence. Le client doit avoir
    /// stocké cette baseline pour pouvoir reconstruire — sinon il ignore
    /// le delta et attend le prochain full.
    pub baseline_server_time: u32,
    /// Bit i set ⇔ le joueur de slot i est inclus dans `dirty_players`.
    /// Un slot non set hérite de la baseline.
    pub dirty_players_bits: u64,
    pub dirty_players: Vec<PlayerState>,
    pub entities: Vec<EntityState>,
    pub dirty_pickups: Vec<PickupState>,
    /// Évènements ce tick — non-baselined : la liste de `current` remplace
    /// celle de la baseline (les events sont des « one-shot »).
    pub events: Vec<ServerEvent>,
}

impl SnapshotDelta {
    /// Encode pour la wire. Format :
    /// ```text
    /// u8   tag = TAG_SNAPSHOT_DELTA
    /// u32  server_time
    /// u32  ack_cmd
    /// u8   client_slot
    /// u32  baseline_server_time
    /// u64  dirty_players_bits
    /// u8   dirty_players_count    (= popcount(dirty_players_bits))
    /// PlayerState[dirty_players_count]   (ordre = bits croissants)
    /// u16  entity_count
    /// EntityState[entity_count]
    /// u16  dirty_pickup_count
    /// PickupState[dirty_pickup_count]
    /// ```
    pub fn encode(&self) -> Result<Vec<u8>, Error> {
        let popcount = self.dirty_players_bits.count_ones() as usize;
        if popcount != self.dirty_players.len() {
            return Err(Error::Network(format!(
                "SnapshotDelta: dirty_players_bits popcount={popcount} != Vec len {}",
                self.dirty_players.len()
            )));
        }
        if self.entities.len() > MAX_ENTITIES_PER_SNAPSHOT {
            return Err(Error::Network(format!(
                "SnapshotDelta: trop d'entités ({} > {})",
                self.entities.len(),
                MAX_ENTITIES_PER_SNAPSHOT
            )));
        }
        let cap = 1
            + 4
            + 4
            + 1
            + 4
            + 8
            + 1
            + popcount * PlayerState::WIRE_SIZE
            + 2
            + self.entities.len() * EntityState::WIRE_SIZE
            + 2
            + self.dirty_pickups.len() * PickupState::WIRE_SIZE;
        let mut buf = BytesMut::with_capacity(cap);
        buf.put_u8(TAG_SNAPSHOT_DELTA);
        buf.put_u32_le(self.server_time);
        buf.put_u32_le(self.ack_cmd);
        buf.put_u8(self.client_slot);
        buf.put_u32_le(self.baseline_server_time);
        buf.put_u64_le(self.dirty_players_bits);
        buf.put_u8(popcount as u8);
        for p in &self.dirty_players {
            p.write(&mut buf);
        }
        buf.put_u16_le(self.entities.len() as u16);
        for e in &self.entities {
            e.write(&mut buf);
        }
        buf.put_u16_le(self.dirty_pickups.len() as u16);
        for p in &self.dirty_pickups {
            p.write(&mut buf);
        }
        write_events(&mut buf, &self.events)?;
        Ok(buf.to_vec())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let mut cur = bytes;
        if cur.remaining() < 1 + 4 + 4 + 1 + 4 + 8 + 1 {
            return Err(Error::Network("SnapshotDelta: header tronqué".into()));
        }
        let tag = cur.get_u8();
        if tag != TAG_SNAPSHOT_DELTA {
            return Err(Error::Network(format!(
                "SnapshotDelta: tag {tag} != {TAG_SNAPSHOT_DELTA}"
            )));
        }
        let server_time = cur.get_u32_le();
        let ack_cmd = cur.get_u32_le();
        let client_slot = cur.get_u8();
        let baseline_server_time = cur.get_u32_le();
        let dirty_players_bits = cur.get_u64_le();
        let popcount_announced = cur.get_u8() as usize;
        let popcount_real = dirty_players_bits.count_ones() as usize;
        if popcount_announced != popcount_real {
            return Err(Error::Network(format!(
                "SnapshotDelta: popcount fil {popcount_announced} ≠ bits {popcount_real}"
            )));
        }
        let mut dirty_players = Vec::with_capacity(popcount_real);
        for _ in 0..popcount_real {
            dirty_players.push(PlayerState::read(&mut cur)?);
        }
        if cur.remaining() < 2 {
            return Err(Error::Network("SnapshotDelta: entity_count tronqué".into()));
        }
        let entity_count = cur.get_u16_le() as usize;
        if entity_count > MAX_ENTITIES_PER_SNAPSHOT {
            return Err(Error::Network(format!(
                "SnapshotDelta: entity_count {entity_count} > max"
            )));
        }
        let mut entities = Vec::with_capacity(entity_count);
        for _ in 0..entity_count {
            entities.push(EntityState::read(&mut cur)?);
        }
        if cur.remaining() < 2 {
            return Err(Error::Network("SnapshotDelta: pickup_count tronqué".into()));
        }
        let dirty_pickup_count = cur.get_u16_le() as usize;
        let mut dirty_pickups = Vec::with_capacity(dirty_pickup_count);
        for _ in 0..dirty_pickup_count {
            dirty_pickups.push(PickupState::read(&mut cur)?);
        }
        let events = read_events(&mut cur)?;
        Ok(Self {
            server_time,
            ack_cmd,
            client_slot,
            baseline_server_time,
            dirty_players_bits,
            dirty_players,
            entities,
            dirty_pickups,
            events,
        })
    }

    /// Reconstruit un [`Snapshot`] complet en appliquant ce delta sur
    /// `baseline`. Le `client_slot` et `ack_cmd` sont ceux du delta
    /// (qui sont per-recipient, pas de la baseline « pinned » globale).
    ///
    /// Gestion des slots :
    /// - bit `i` set dans `dirty_players_bits` : on prend `PlayerState` du delta
    /// - bit `i` non set, mais joueur `i` dans `baseline.players` : on hérite
    /// - bit `i` non set, baseline absente : le slot n'existe pas
    pub fn apply_to_baseline(&self, baseline: &Snapshot) -> Snapshot {
        let mut by_slot: [Option<PlayerState>; 64] = [None; 64];
        for p in &baseline.players {
            if (p.slot as usize) < 64 {
                by_slot[p.slot as usize] = Some(*p);
            }
        }
        // Override avec les dirty — l'ordre dans `dirty_players` correspond
        // à l'ordre des bits dans `dirty_players_bits` (du LSB au MSB).
        let mut dirty_iter = self.dirty_players.iter();
        for i in 0..64 {
            if self.dirty_players_bits & (1u64 << i) != 0 {
                if let Some(p) = dirty_iter.next() {
                    by_slot[i] = Some(*p);
                }
            }
        }
        let players: Vec<PlayerState> = by_slot.into_iter().flatten().collect();

        // Pickups : baseline → HashMap, override par dirty_pickups.
        let mut pickup_map: std::collections::HashMap<u16, u8> = baseline
            .pickups
            .iter()
            .map(|p| (p.id, p.available))
            .collect();
        for p in &self.dirty_pickups {
            pickup_map.insert(p.id, p.available);
        }
        let mut pickups: Vec<PickupState> = pickup_map
            .into_iter()
            .map(|(id, available)| PickupState { id, available })
            .collect();
        pickups.sort_by_key(|p| p.id);

        Snapshot {
            server_time: self.server_time,
            ack_cmd: self.ack_cmd,
            client_slot: self.client_slot,
            players,
            entities: self.entities.clone(),
            pickups,
            // Events viennent du delta (pas baselinés).
            events: self.events.clone(),
            // players_info hérité de la baseline — change rarement,
            // refresh forcé à chaque full (FULL_SNAPSHOT_INTERVAL).
            players_info: baseline.players_info.clone(),
        }
    }

    /// Calcule le delta `current` − `baseline`. Un `PlayerState` est
    /// « dirty » s'il diffère bit-à-bit de celui de baseline (ou s'il
    /// n'existait pas dans la baseline).
    ///
    /// Note : on indexe par `slot` côté baseline ET côté current. Si un
    /// même slot a une `PlayerState` différente, il devient dirty.
    pub fn compute_diff(baseline: &Snapshot, current: &Snapshot) -> Self {
        // Index baseline par slot.
        let mut base_by_slot: [Option<PlayerState>; 64] = [None; 64];
        for p in &baseline.players {
            if (p.slot as usize) < 64 {
                base_by_slot[p.slot as usize] = Some(*p);
            }
        }

        let mut dirty_bits: u64 = 0;
        let mut dirty_players: Vec<PlayerState> = Vec::new();
        // Itération dans l'ordre slot croissant pour matcher l'ordre des
        // bits LSB → MSB attendu par `apply_to_baseline`.
        let mut current_by_slot: [Option<PlayerState>; 64] = [None; 64];
        for p in &current.players {
            if (p.slot as usize) < 64 {
                current_by_slot[p.slot as usize] = Some(*p);
            }
        }
        for i in 0..64 {
            if let Some(cur) = current_by_slot[i] {
                let changed = match base_by_slot[i] {
                    Some(b) => b != cur,
                    None => true,
                };
                if changed {
                    dirty_bits |= 1u64 << i;
                    dirty_players.push(cur);
                }
            }
        }

        // Pickups : seuls les changements (id présent dans current avec
        // available différent, ou absent de baseline).
        let mut base_pick: std::collections::HashMap<u16, u8> = baseline
            .pickups
            .iter()
            .map(|p| (p.id, p.available))
            .collect();
        let mut dirty_pickups: Vec<PickupState> = Vec::new();
        for p in &current.pickups {
            match base_pick.remove(&p.id) {
                Some(b) if b == p.available => {}
                _ => dirty_pickups.push(*p),
            }
        }

        Self {
            server_time: current.server_time,
            ack_cmd: current.ack_cmd,
            client_slot: current.client_slot,
            baseline_server_time: baseline.server_time,
            dirty_players_bits: dirty_bits,
            dirty_players,
            entities: current.entities.clone(),
            dirty_pickups,
            // Events ne sont jamais hérités de la baseline — ils sont
            // « 1-shot ». On prend ceux du current tel quel.
            events: current.events.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_angle_roundtrip_within_precision() {
        for &deg in &[0.0_f32, 45.0, 90.0, 179.0, 180.0, 270.0, 359.0] {
            let q = UserCmd::quantize_angle(deg);
            let back = UserCmd::dequantize_angle(q);
            // Précision attendue ~0.006° ; on accepte 0.01°.
            assert!(
                (back - deg).abs() < 0.01,
                "deg={deg}, back={back}, q={q}"
            );
        }
    }

    #[test]
    fn quantize_angle_handles_wraparound() {
        // -90° doit être équivalent à 270° après quantification.
        let qa = UserCmd::quantize_angle(-90.0);
        let qb = UserCmd::quantize_angle(270.0);
        assert_eq!(qa, qb);
    }

    #[test]
    fn quantize_axis_extremes() {
        assert_eq!(UserCmd::quantize_axis(1.0), 127);
        assert_eq!(UserCmd::quantize_axis(-1.0), -127);
        assert_eq!(UserCmd::quantize_axis(0.0), 0);
        // Au-delà de 1.0 → clamp.
        assert_eq!(UserCmd::quantize_axis(5.0), 127);
    }

    #[test]
    fn usercmd_roundtrip() {
        let cmd = UserCmd {
            cmd_number: 42,
            forward: 100,
            side: -50,
            up: 0,
            buttons: buttons::FIRE | buttons::JUMP,
            view_pitch: UserCmd::quantize_angle(-15.0),
            view_yaw: UserCmd::quantize_angle(180.0),
            view_roll: 0,
            delta_ms: 8,
            weapon: 5,
        };
        let mut buf = BytesMut::new();
        cmd.write(&mut buf);
        assert_eq!(buf.len(), UserCmd::WIRE_SIZE);
        let mut slice: &[u8] = &buf;
        let back = UserCmd::read(&mut slice).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn client_packet_roundtrip_multi_cmds() {
        let mut cmds = Vec::new();
        for i in 0..5 {
            cmds.push(UserCmd {
                cmd_number: 100 + i,
                forward: i as i8,
                ..Default::default()
            });
        }
        let pkt = ClientPacket {
            server_time_ack: 12345,
            cmds,
        };
        let bytes = pkt.encode().unwrap();
        let back = ClientPacket::decode(&bytes).unwrap();
        assert_eq!(pkt, back);
    }

    #[test]
    fn client_packet_rejects_overflow() {
        let pkt = ClientPacket {
            server_time_ack: 0,
            cmds: vec![UserCmd::default(); MAX_USERCMDS_PER_PACKET + 1],
        };
        assert!(pkt.encode().is_err());
    }

    #[test]
    fn client_packet_rejects_bad_tag() {
        let mut bytes = vec![0u8; 16];
        bytes[0] = 99; // tag inconnu
        assert!(ClientPacket::decode(&bytes).is_err());
    }

    #[test]
    fn snapshot_roundtrip_full() {
        let snap = Snapshot {
            server_time: 99_000,
            ack_cmd: 7,
            client_slot: 2,
            players: vec![
                PlayerState {
                    slot: 0,
                    flags: player_flags::ON_GROUND,
                    health: 100,
                    armor: 50,
                    weapon: 3,
                    powerups: 0,
                    frags: 4,
                    deaths: 1,
                    origin: [10.0, 20.0, 30.0],
                    velocity: [100.0, 0.0, 0.0],
                    view_angles: [0.0, 90.0, 0.0],
                    ammo: [0, 0, 50, 10, 0, 5, 0, 0, 30, 0],
                    team: 0,
                },
                PlayerState {
                    slot: 1,
                    flags: player_flags::BOT | player_flags::ON_GROUND,
                    health: 75,
                    armor: 0,
                    weapon: 5,
                    powerups: 1, // Quad
                    frags: 2,
                    deaths: 3,
                    origin: [-50.0, 0.0, 32.0],
                    velocity: [0.0, 200.0, 0.0],
                    view_angles: [10.0, 270.0, 0.0],
                    ammo: [0; 10],
                    team: 0,
                },
            ],
            entities: vec![EntityState {
                id: 1234,
                kind: EntityKindWire::Rocket,
                owner: 0,
                origin: [0.0, 0.0, 100.0],
                velocity: [900.0, 0.0, 0.0],
            }],
            pickups: vec![
                PickupState {
                    id: 5,
                    available: 0,
                },
                PickupState {
                    id: 12,
                    available: 1,
                },
            ],
            events: vec![],
            players_info: vec![],
        };
        let bytes = snap.encode().unwrap();
        let back = Snapshot::decode(&bytes).unwrap();
        assert_eq!(snap, back);
    }

    #[test]
    fn snapshot_empty_is_valid() {
        let snap = Snapshot {
            server_time: 1,
            ack_cmd: 0,
            client_slot: 0,
            players: vec![],
            entities: vec![],
            pickups: vec![],
            events: vec![],
            players_info: vec![],
        };
        let bytes = snap.encode().unwrap();
        let back = Snapshot::decode(&bytes).unwrap();
        assert_eq!(snap, back);
    }

    #[test]
    fn snapshot_rejects_truncated() {
        let mut bytes = Snapshot {
            server_time: 1,
            ack_cmd: 0,
            client_slot: 0,
            players: vec![PlayerState::default()],
            entities: vec![],
            pickups: vec![],
            events: vec![],
            players_info: vec![],
        }
        .encode()
        .unwrap();
        // Coupe au milieu du PlayerState.
        bytes.truncate(bytes.len() - 5);
        assert!(Snapshot::decode(&bytes).is_err());
    }

    fn make_player(slot: u8, x: f32, frags: i16) -> PlayerState {
        PlayerState {
            slot,
            flags: player_flags::ON_GROUND,
            health: 100,
            armor: 0,
            weapon: 2,
            powerups: 0,
            frags,
            deaths: 0,
            origin: [x, 0.0, 0.0],
            velocity: [0.0; 3],
            view_angles: [0.0; 3],
            ammo: [0; 10],
            team: 0,
        }
    }

    #[test]
    fn snapshot_delta_roundtrip_via_wire() {
        let delta = SnapshotDelta {
            server_time: 1000,
            ack_cmd: 42,
            client_slot: 1,
            baseline_server_time: 950,
            // Slots 0 et 3 dirty.
            dirty_players_bits: (1 << 0) | (1 << 3),
            dirty_players: vec![make_player(0, 10.0, 1), make_player(3, 30.0, 2)],
            entities: vec![],
            dirty_pickups: vec![PickupState { id: 7, available: 0 }],
            events: vec![],
        };
        let bytes = delta.encode().unwrap();
        let back = SnapshotDelta::decode(&bytes).unwrap();
        assert_eq!(delta, back);
    }

    #[test]
    fn snapshot_delta_apply_replaces_dirty_keeps_others() {
        // Baseline : 4 joueurs.
        let baseline = Snapshot {
            server_time: 100,
            ack_cmd: 0,
            client_slot: 0,
            players: vec![
                make_player(0, 0.0, 0),
                make_player(1, 100.0, 0),
                make_player(2, 200.0, 0),
                make_player(3, 300.0, 0),
            ],
            entities: vec![],
            pickups: vec![PickupState { id: 1, available: 1 }],
            events: vec![],
            players_info: vec![],
        };
        // Delta : seuls les slots 1 et 3 ont bougé.
        let delta = SnapshotDelta {
            server_time: 150,
            ack_cmd: 5,
            client_slot: 0,
            baseline_server_time: 100,
            dirty_players_bits: (1 << 1) | (1 << 3),
            dirty_players: vec![make_player(1, 150.0, 1), make_player(3, 333.0, 0)],
            entities: vec![],
            dirty_pickups: vec![],
            events: vec![],
        };
        let recon = delta.apply_to_baseline(&baseline);
        // Slots 0 et 2 doivent venir de la baseline.
        assert_eq!(recon.players.len(), 4);
        assert_eq!(recon.players[0].origin[0], 0.0);
        assert_eq!(recon.players[1].origin[0], 150.0);
        assert_eq!(recon.players[1].frags, 1);
        assert_eq!(recon.players[2].origin[0], 200.0);
        assert_eq!(recon.players[3].origin[0], 333.0);
        // Pickups inchangés.
        assert_eq!(recon.pickups.len(), 1);
        assert_eq!(recon.pickups[0], PickupState { id: 1, available: 1 });
        // Les champs « per-recipient » viennent du delta.
        assert_eq!(recon.server_time, 150);
        assert_eq!(recon.ack_cmd, 5);
    }

    #[test]
    fn snapshot_delta_diff_then_apply_is_identity() {
        let baseline = Snapshot {
            server_time: 1,
            ack_cmd: 0,
            client_slot: 0,
            players: vec![
                make_player(0, 0.0, 0),
                make_player(1, 100.0, 0),
            ],
            entities: vec![],
            pickups: vec![],
            events: vec![],
            players_info: vec![],
        };
        let current = Snapshot {
            server_time: 2,
            ack_cmd: 3,
            client_slot: 0,
            players: vec![
                make_player(0, 5.0, 0),     // bougé
                make_player(1, 100.0, 0),   // identique → ne devrait PAS être dirty
                make_player(2, 50.0, 0),    // nouveau → dirty
            ],
            entities: vec![EntityState {
                id: 7,
                kind: EntityKindWire::Rocket,
                owner: 0,
                origin: [0.0; 3],
                velocity: [900.0, 0.0, 0.0],
            }],
            pickups: vec![],
            events: vec![],
            players_info: vec![],
        };
        let delta = SnapshotDelta::compute_diff(&baseline, &current);
        // Bits attendus : 0 (changé) et 2 (nouveau). Pas le 1.
        assert_eq!(delta.dirty_players_bits, (1 << 0) | (1 << 2));
        assert_eq!(delta.dirty_players.len(), 2);

        // Apply doit reconstruire identiquement.
        let recon = delta.apply_to_baseline(&baseline);
        assert_eq!(recon.players.len(), current.players.len());
        for (a, b) in recon.players.iter().zip(current.players.iter()) {
            assert_eq!(a, b);
        }
        assert_eq!(recon.entities, current.entities);
    }

    #[test]
    fn server_event_explosion_roundtrip() {
        let evt = ServerEvent::Explosion {
            pos: [123.0, -45.5, 678.25],
            kind: ExplosionKind::Rocket,
        };
        let mut buf = BytesMut::new();
        evt.write(&mut buf);
        let mut slice: &[u8] = &buf;
        let back = ServerEvent::read(&mut slice).unwrap();
        assert_eq!(evt, back);
    }

    #[test]
    fn server_event_rail_trail_roundtrip() {
        let evt = ServerEvent::RailTrail {
            from: [10.0, 20.0, 30.0],
            to: [-100.5, 200.25, 50.0],
            owner: 4,
        };
        let mut buf = BytesMut::new();
        evt.write(&mut buf);
        let mut slice: &[u8] = &buf;
        assert_eq!(ServerEvent::read(&mut slice).unwrap(), evt);
    }

    #[test]
    fn server_event_match_ended_roundtrip() {
        for winner in [0u8, 7, 255] {
            let evt = ServerEvent::MatchEnded { winner };
            let mut buf = BytesMut::new();
            evt.write(&mut buf);
            let mut slice: &[u8] = &buf;
            assert_eq!(ServerEvent::read(&mut slice).unwrap(), evt);
        }
    }

    #[test]
    fn server_event_chat_roundtrip_and_helpers() {
        let evt = ServerEvent::new_chat(3, "hello world ! 👋");
        let mut buf = BytesMut::new();
        evt.write(&mut buf);
        let mut slice: &[u8] = &buf;
        let back = ServerEvent::read(&mut slice).unwrap();
        assert_eq!(evt, back);
        let (s, m) = back.chat_message().unwrap();
        assert_eq!(s, 3);
        assert!(m.starts_with("hello world"));
    }

    #[test]
    fn server_event_chat_truncates_to_96() {
        let long = "x".repeat(200);
        let evt = ServerEvent::new_chat(0, &long);
        let (_, m) = evt.chat_message().unwrap();
        assert_eq!(m.len(), 96);
    }

    #[test]
    fn server_event_match_started_roundtrip() {
        let evt = ServerEvent::MatchStarted;
        let mut buf = BytesMut::new();
        evt.write(&mut buf);
        // 1 octet (tag uniquement, pas de payload).
        assert_eq!(buf.len(), 1);
        let mut slice: &[u8] = &buf;
        assert_eq!(ServerEvent::read(&mut slice).unwrap(), evt);
    }

    #[test]
    fn server_event_player_killed_roundtrip() {
        let evt = ServerEvent::PlayerKilled {
            victim: 3,
            killer: 1,
            weapon: 5,
        };
        let mut buf = BytesMut::new();
        evt.write(&mut buf);
        let mut slice: &[u8] = &buf;
        assert_eq!(ServerEvent::read(&mut slice).unwrap(), evt);
    }

    #[test]
    fn snapshot_with_events_roundtrips() {
        let snap = Snapshot {
            server_time: 50,
            ack_cmd: 0,
            client_slot: 0,
            players: vec![],
            entities: vec![],
            pickups: vec![],
            events: vec![
                ServerEvent::Explosion {
                    pos: [10.0, 20.0, 30.0],
                    kind: ExplosionKind::Plasma,
                },
                ServerEvent::PlayerKilled {
                    victim: 2,
                    killer: 0,
                    weapon: 6,
                },
            ],
            players_info: vec![],
        };
        let bytes = snap.encode().unwrap();
        let back = Snapshot::decode(&bytes).unwrap();
        assert_eq!(snap, back);
        assert_eq!(back.events.len(), 2);
    }

    /// Les events ne sont PAS hérités de la baseline — un delta n'inclut
    /// que ses propres events, pas ceux d'un snapshot précédent.
    #[test]
    fn delta_apply_takes_events_from_delta_only() {
        let baseline = Snapshot {
            server_time: 1,
            ack_cmd: 0,
            client_slot: 0,
            players: vec![make_player(0, 0.0, 0)],
            entities: vec![],
            pickups: vec![],
            events: vec![ServerEvent::Explosion {
                pos: [1.0, 2.0, 3.0],
                kind: ExplosionKind::Rocket,
            }],
            players_info: vec![],
        };
        let delta = SnapshotDelta {
            server_time: 2,
            ack_cmd: 0,
            client_slot: 0,
            baseline_server_time: 1,
            dirty_players_bits: 0,
            dirty_players: vec![],
            entities: vec![],
            dirty_pickups: vec![],
            events: vec![],
        };
        let recon = delta.apply_to_baseline(&baseline);
        assert!(recon.events.is_empty(), "events de baseline ne doivent pas fuiter");
    }

    #[test]
    fn snapshot_delta_rejects_popcount_mismatch() {
        // dirty_players_bits dit 3 bits, mais la Vec n'en a que 2.
        let bad = SnapshotDelta {
            server_time: 1,
            ack_cmd: 0,
            client_slot: 0,
            baseline_server_time: 0,
            dirty_players_bits: 0b111,
            dirty_players: vec![make_player(0, 0.0, 0), make_player(1, 0.0, 0)],
            entities: vec![],
            dirty_pickups: vec![],
            events: vec![],
        };
        assert!(bad.encode().is_err());
    }

    #[test]
    fn snapshot_rejects_unknown_entity_kind() {
        let mut bytes = Snapshot {
            server_time: 1,
            ack_cmd: 0,
            client_slot: 0,
            players: vec![],
            entities: vec![EntityState {
                id: 1,
                kind: EntityKindWire::Rocket,
                owner: 0,
                origin: [0.0; 3],
                velocity: [0.0; 3],
            }],
            pickups: vec![],
            events: vec![],
            players_info: vec![],
        }
        .encode()
        .unwrap();
        // L'octet `kind` est juste après tag(1) + server_time(4) + ack(4)
        // + slot(1) + player_count(2) + entity_count(2) + id(4).
        let kind_offset = 1 + 4 + 4 + 1 + 2 + 2 + 4;
        bytes[kind_offset] = 99;
        assert!(Snapshot::decode(&bytes).is_err());
    }
}
