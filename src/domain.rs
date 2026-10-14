use std::fmt;

use anyhow::{Result, bail};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionMethod {
    SampledFrames,
    SemanticVideo,
}

impl fmt::Display for SelectionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SampledFrames => "sampled_frames",
            Self::SemanticVideo => "semantic_video",
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GameInput {
    Title(String),
    Context(String),
}

impl GameInput {
    pub fn new(title: Option<String>, context: Option<String>) -> Result<Self> {
        match (title, context) {
            (Some(title), None) if !title.trim().is_empty() => {
                Ok(Self::Title(title.trim().to_owned()))
            }
            (None, Some(context)) if !context.trim().is_empty() => {
                Ok(Self::Context(context.trim().to_owned()))
            }
            (Some(_), Some(_)) => {
                bail!("specify exactly one of --game-title and --game-context")
            }
            (None, None) => bail!("specify --game-title or --game-context"),
            _ => bail!("--game-title and --game-context must not be blank"),
        }
    }
}
