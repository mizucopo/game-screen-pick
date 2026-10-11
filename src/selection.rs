//! Deterministic quality and diversity selection, independent of AI transport.
use crate::quality::{NEAR_DUPLICATE_DISTANCE, Quality};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assessment {
    pub id: String,
    pub blog_score: f64,
    pub transition: bool,
    pub scene: String,
    pub reason: String,
    /// Explicit semantic classification; scene labels themselves may be any language.
    pub special: bool,
}

impl Assessment {
    pub fn rejection(&self) -> Option<&'static str> {
        if self.transition {
            Some("transition/context rejection")
        } else if self.blog_score < 20.0 {
            Some("low blog usefulness")
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticEvent {
    pub source: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub representative_seconds: f64,
    pub importance: f64,
    pub scene: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: String,
    pub source: String,
    pub requested_seconds: f64,
    pub actual_seconds: f64,
    pub duration_seconds: f64,
    pub image: PathBuf,
    pub quality: Quality,
    pub event: Option<SemanticEvent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Exclusion {
    pub id: String,
    pub reason: String,
}

#[derive(Debug)]
pub struct Selection {
    /// Indices into the supplied candidates, in final rank order.
    pub indices: Vec<usize>,
    pub exclusions: Vec<Exclusion>,
    pub notes: Vec<String>,
}

pub const MAX_CANDIDATES: usize = 4096;

pub fn sample_positions(duration: f64, count: usize) -> Result<Vec<f64>> {
    ensure!(
        duration.is_finite() && duration > 0.0,
        "invalid sampling duration"
    );
    ensure!(
        (1..=MAX_CANDIDATES).contains(&count),
        "sample count must be 1..=4096"
    );
    Ok(if count == 1 {
        vec![duration / 2.0]
    } else {
        (0..count)
            .map(|index| duration * index as f64 / (count - 1) as f64)
            .collect()
    })
}

pub fn event_positions(event: &SemanticEvent, duration: f64) -> Result<Vec<f64>> {
    ensure!(
        duration.is_finite()
            && duration > 0.0
            && event.start_seconds.is_finite()
            && event.end_seconds.is_finite()
            && event.representative_seconds.is_finite()
            && event.start_seconds >= 0.0
            && event.end_seconds <= duration
            && event.start_seconds < event.end_seconds
            && (event.start_seconds..=event.end_seconds).contains(&event.representative_seconds)
            && event.importance.is_finite()
            && (0.0..=100.0).contains(&event.importance)
            && !event.source.trim().is_empty()
            && !event.scene.trim().is_empty()
            && !event.reason.trim().is_empty(),
        "semantic event has invalid source, interval, importance or evidence"
    );
    let mut positions = vec![
        (event.start_seconds + event.representative_seconds) / 2.0,
        event.representative_seconds,
        (event.representative_seconds + event.end_seconds) / 2.0,
    ];
    positions.sort_by(f64::total_cmp);
    positions.dedup();
    Ok(positions)
}

pub fn validate_assessments(candidates: &[Candidate], assessments: &[Assessment]) -> Result<()> {
    ensure!(
        candidates.len() <= MAX_CANDIDATES,
        "candidate count exceeds 4096"
    );
    let mut ids = BTreeSet::new();
    for candidate in candidates {
        ensure!(
            !candidate.id.trim().is_empty()
                && candidate.id.len() <= 32
                && !candidate.id.chars().any(char::is_control)
                && ids.insert(candidate.id.as_str())
                && !candidate.source.trim().is_empty()
                && candidate.requested_seconds.is_finite()
                && candidate.actual_seconds.is_finite()
                && candidate.duration_seconds.is_finite()
                && candidate.duration_seconds > 0.0
                && (0.0..=candidate.duration_seconds).contains(&candidate.requested_seconds)
                && (0.0..=candidate.duration_seconds).contains(&candidate.actual_seconds)
                && candidate.quality.score.is_finite()
                && (0.0..=100.0).contains(&candidate.quality.score),
            "invalid or duplicate candidate identity, time or quality"
        );
        if let Some(event) = &candidate.event {
            event_positions(event, candidate.duration_seconds)?;
            ensure!(
                event.source == candidate.source,
                "semantic event source differs from candidate"
            );
            ensure!(
                (event.start_seconds..=event.end_seconds).contains(&candidate.actual_seconds),
                "candidate actual time lies outside its semantic event"
            );
        }
    }
    ensure!(
        assessments.len() == candidates.len(),
        "assessment ID count differs from requested candidates"
    );
    for assessment in assessments {
        ensure!(
            ids.remove(assessment.id.as_str()),
            "unknown or duplicate assessment ID"
        );
        ensure!(
            assessment.blog_score.is_finite()
                && (0.0..=100.0).contains(&assessment.blog_score)
                && !assessment.scene.trim().is_empty()
                && assessment.scene.len() <= 512
                && !assessment.reason.trim().is_empty()
                && assessment.reason.len() <= 4096
                && !assessment.scene.chars().any(char::is_control)
                && !assessment.reason.chars().any(char::is_control),
            "assessment score, scene or evidence is invalid"
        );
    }
    ensure!(ids.is_empty(), "missing assessment ID");
    Ok(())
}

fn scene(assessment: &Assessment) -> String {
    assessment
        .scene
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn time_band(candidate: &Candidate) -> u8 {
    ((candidate.actual_seconds / candidate.duration_seconds * 3.0).floor() as u8).min(2)
}

fn merit(candidate: &Candidate, assessment: &Assessment) -> f64 {
    0.8 * assessment.blog_score
        + 0.2 * candidate.quality.score
        + candidate
            .event
            .as_ref()
            .map_or(0.0, |event| 0.15 * event.importance)
}

/// Stable tie-break: merit, source filename bytes, actual PTS, requested time,
/// display ID. Callback order and hash iteration never affect ranking.
fn tie(
    left: usize,
    right: usize,
    candidates: &[Candidate],
    ratings: &[&Assessment],
) -> std::cmp::Ordering {
    merit(&candidates[right], ratings[right])
        .total_cmp(&merit(&candidates[left], ratings[left]))
        .then_with(|| candidates[left].source.cmp(&candidates[right].source))
        .then_with(|| {
            candidates[left]
                .actual_seconds
                .total_cmp(&candidates[right].actual_seconds)
        })
        .then_with(|| {
            candidates[left]
                .requested_seconds
                .total_cmp(&candidates[right].requested_seconds)
        })
        .then_with(|| candidates[left].id.cmp(&candidates[right].id))
}

pub fn select(
    candidates: &[Candidate],
    assessments: &[Assessment],
    count: usize,
) -> Result<Selection> {
    ensure!((1..=999).contains(&count), "selected count must be 1..=999");
    let selected = select_up_to(candidates, assessments, count)?;
    let mut reasons = BTreeMap::new();
    for exclusion in &selected.exclusions {
        *reasons.entry(exclusion.reason.as_str()).or_insert(0) += 1;
    }
    ensure!(
        selected.indices.len() == count,
        "insufficient valid distinct candidates: requested {count}, available {}; exclusions {reasons:?}; sample more frames or provide more recordings",
        selected.indices.len()
    );
    Ok(selected)
}

/// Best available subset for replenishment/secondary planning. The final caller
/// must use select, which fails rather than filling missing slots with rejects.
pub fn select_up_to(
    candidates: &[Candidate],
    assessments: &[Assessment],
    count: usize,
) -> Result<Selection> {
    ensure!(
        (1..=MAX_CANDIDATES).contains(&count),
        "shortlist count must be 1..=4096"
    );
    validate_assessments(candidates, assessments)?;
    let by_id: BTreeMap<_, _> = assessments
        .iter()
        .map(|assessment| (assessment.id.as_str(), assessment))
        .collect();
    let ratings: Vec<_> = candidates
        .iter()
        .map(|candidate| by_id[candidate.id.as_str()])
        .collect();
    let scene_names: Vec<_> = ratings.iter().map(|assessment| scene(assessment)).collect();
    let mut result = Selection {
        indices: Vec::new(),
        exclusions: Vec::new(),
        notes: Vec::new(),
    };
    let mut available = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let rejection = candidate
            .quality
            .rejection
            .clone()
            .or_else(|| ratings[index].rejection().map(str::to_owned));
        if let Some(reason) = rejection {
            result.exclusions.push(Exclusion {
                id: candidate.id.clone(),
                reason,
            });
        } else {
            available.push(index);
        }
    }
    let mut sources = BTreeSet::new();
    let mut scenes = BTreeSet::new();
    let mut bands = BTreeSet::new();
    let mut special_count = 0;
    let special_limit = if count == 1 { 0 } else { (count / 3).max(1) };
    let mut nearest = vec![1.0_f64; candidates.len()];
    let mut special_blocked = vec![false; candidates.len()];
    while !available.is_empty() && result.indices.len() < count {
        // Remove duplicates only against chosen images, so the best useful
        // representative survives even when it arrived later in a callback.
        available.retain(|&index| {
            if let Some(&chosen) = result.indices.last() {
                nearest[index] = nearest[index].min(
                    candidates[index]
                        .quality
                        .distance(&candidates[chosen].quality),
                );
            }
            if nearest[index] < NEAR_DUPLICATE_DISTANCE {
                result.exclusions.push(Exclusion {
                    id: candidates[index].id.clone(),
                    reason: "near duplicate of selected image".into(),
                });
                false
            } else {
                true
            }
        });
        if available.is_empty() {
            break;
        }
        let normal_available = available.iter().any(|&index| !ratings[index].special);
        let permitted: Vec<_> = available
            .iter()
            .copied()
            .filter(|&index| {
                special_blocked[index] = ratings[index].special
                    && normal_available
                    && (special_count >= special_limit || result.indices.is_empty());
                !special_blocked[index]
            })
            .collect();
        // Coverage is lexicographic: unseen source, scene, source time-third,
        // then usefulness with a visual-distance bonus. No quota demands absent scenes.
        let coverage = |index: usize| {
            let candidate = &candidates[index];
            (
                !sources.contains(candidate.source.as_str()),
                !scenes.contains(&scene_names[index]),
                !bands.contains(&(candidate.source.as_str(), time_band(candidate))),
            )
        };
        let best = *permitted
            .iter()
            .min_by(|&&left, &&right| {
                coverage(right).cmp(&coverage(left)).then_with(|| {
                    let value = |index: usize| {
                        merit(&candidates[index], ratings[index]) + 20.0 * nearest[index]
                    };
                    value(right)
                        .total_cmp(&value(left))
                        .then_with(|| tie(left, right, candidates, &ratings))
                })
            })
            .expect("at least one permissible candidate");
        sources.insert(candidates[best].source.as_str());
        scenes.insert(scene_names[best].clone());
        bands.insert((
            candidates[best].source.as_str(),
            time_band(&candidates[best]),
        ));
        special_count += usize::from(ratings[best].special);
        result.indices.push(best);
        available.retain(|&index| index != best);
    }
    for index in available {
        let reason = if result.indices.iter().any(|&chosen| {
            candidates[index]
                .quality
                .near_duplicate(&candidates[chosen].quality)
        }) {
            "near duplicate of selected image"
        } else if special_blocked[index] {
            "special screen limited to favor normal progress"
        } else {
            "not selected by diversity and usefulness ranking"
        };
        result.exclusions.push(Exclusion {
            id: candidates[index].id.clone(),
            reason: reason.into(),
        });
    }
    let valid_sources: BTreeSet<_> = candidates
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            candidate.quality.rejection.is_none() && ratings[*index].rejection().is_none()
        })
        .map(|(_, candidate)| &candidate.source)
        .collect();
    if sources.len() < valid_sources.len() {
        result.notes.push(format!("source coverage {}/{}: limited slots, near duplicates or preference for normal progress",sources.len(),valid_sources.len()));
    }
    result.notes.push(format!(
        "{} normal progress / {} special screens; {} scene(s), {} source/time band(s)",
        result.indices.len() - special_count,
        special_count,
        scenes.len(),
        bands.len()
    ));
    result
        .exclusions
        .sort_by(|left, right| left.id.cmp(&right.id));
    Ok(result)
}
