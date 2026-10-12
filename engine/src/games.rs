//! The game manifest (games/manifest.json, written by chipformer/games.py).

use serde_json::Value;

pub struct Game {
    pub id: String,
    pub title: String,
    pub file: String, // relative to the manifest's directory
    pub ipf: usize,
    pub quirks: String, // comma-separated flags
    pub keys: Vec<(String, usize)>, // role ("left", "a", "player2Up", ...) -> CHIP-8 key
    pub controls: String,
    /// roles delivered as one short pulse per press (see chipformer/games.py TUNING)
    pub tap: Vec<String>,
}

/// Length of a tap pulse in 60 Hz frames.
pub const TAP_FRAMES: u32 = 3;

pub fn parse_manifest(text: &str) -> Result<Vec<Game>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("manifest: {e}"))?;
    let games = v["games"].as_array().ok_or("manifest has no games")?;
    Ok(games
        .iter()
        .map(|g| Game {
            id: g["id"].as_str().unwrap_or_default().to_string(),
            title: g["title"].as_str().unwrap_or_default().to_string(),
            file: g["file"].as_str().unwrap_or_default().to_string(),
            ipf: g["ipf"].as_u64().unwrap_or(15) as usize,
            quirks: g["quirks"]
                .as_array()
                .map(|q| q.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(","))
                .unwrap_or_default(),
            keys: g["keys"]
                .as_object()
                .map(|k| k.iter().filter_map(|(r, c)| Some((r.clone(), c.as_u64()? as usize))).collect())
                .unwrap_or_default(),
            controls: g["controls"].as_str().unwrap_or_default().to_string(),
            tap: g["tap"]
                .as_array()
                .map(|t| t.iter().filter_map(|r| r.as_str().map(str::to_string)).collect())
                .unwrap_or_default(),
        })
        .collect())
}

/// Keyboard keys (by name) for a key role. Two-player games put player 1 on W/S and A/D and
/// player 2 on the arrow keys; one-player games accept both arrows and WASD.
pub fn role_keys(role: &str, two_player: bool) -> &'static [&'static str] {
    match (role, two_player) {
        ("up", true) => &["W", "A"],
        ("down", true) => &["S", "D"],
        ("player2Up", _) => &["Up", "Left"],
        ("player2Down", _) => &["Down", "Right"],
        ("up", false) => &["Up", "W"],
        ("down", false) => &["Down", "S"],
        ("left", _) => &["Left", "A"],
        ("right", _) => &["Right", "D"],
        ("a", _) => &["Space", "E"],
        ("b", _) => &["Enter", "Q"],
        _ => &[],
    }
}

impl Game {
    pub fn two_player(&self) -> bool {
        self.keys.iter().any(|(r, _)| r.starts_with("player2"))
    }

    /// CHIP-8 keys that are tapped rather than held.
    pub fn tap_chips(&self) -> Vec<usize> {
        self.keys.iter().filter(|(r, _)| self.tap.contains(r)).map(|(_, c)| *c).collect()
    }

    /// (keyboard key name, CHIP-8 key) pairs for this game's roles.
    pub fn bindings(&self) -> Vec<(&'static str, usize)> {
        let two = self.two_player();
        self.keys
            .iter()
            .flat_map(|(role, chip)| role_keys(role, two).iter().map(move |k| (*k, *chip)))
            .collect()
    }
}
