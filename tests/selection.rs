use game_screen_pick::{
    quality::analyze,
    selection::{
        Assessment, Candidate, SemanticEvent, event_positions, sample_positions, select,
        select_up_to, validate_assessments,
    },
};
use image::{Rgb, RgbImage};

fn candidates(directory: &std::path::Path, count: usize) -> (Vec<Candidate>, Vec<Assessment>) {
    let mut candidates = Vec::new();
    let mut ratings = Vec::new();
    for index in 0..count {
        let image = directory.join(format!("{index}.png"));
        RgbImage::from_fn(96, 64, |x, y| {
            let tile = (x / 8 + y / 8 * 12) as usize;
            let value = ((tile * 73 + index * 47) % 256) as u8;
            Rgb([
                value,
                value.wrapping_add((index * 23) as u8),
                value.wrapping_add(96),
            ])
        })
        .save(&image)
        .unwrap();
        candidates.push(Candidate {
            id: format!("F{index:04}"),
            source: "録画.mkv".into(),
            requested_seconds: index as f64,
            actual_seconds: index as f64,
            duration_seconds: 30.0,
            quality: analyze(&image).unwrap(),
            image,
            event: None,
        });
        ratings.push(Assessment {
            id: format!("F{index:04}"),
            blog_score: 80.0,
            transition: false,
            scene: "探索".into(),
            reason: "鮮明な通常進行".into(),
            special: false,
        });
    }
    (candidates, ratings)
}

#[test]
fn positions_cover_endpoints_and_semantic_neighborhood_with_evidence() {
    assert_eq!(sample_positions(9.0, 4).unwrap(), [0.0, 3.0, 6.0, 9.0]);
    assert_eq!(sample_positions(9.0, 1).unwrap(), [4.5]);
    for invalid in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        assert!(sample_positions(invalid, 2).is_err());
    }
    assert!(sample_positions(10.0, 4097).is_err());
    let event = SemanticEvent {
        source: "録画.mkv".into(),
        start_seconds: 2.0,
        end_seconds: 6.0,
        representative_seconds: 4.0,
        importance: 85.0,
        scene: "会話".into(),
        reason: "物語の転機".into(),
    };
    assert_eq!(event_positions(&event, 10.0).unwrap(), [3.0, 4.0, 5.0]);
    let mut invalid = event.clone();
    invalid.end_seconds = 11.0;
    assert!(event_positions(&invalid, 10.0).is_err());
    invalid = event.clone();
    invalid.representative_seconds = 1.0;
    assert!(event_positions(&invalid, 10.0).is_err());
    invalid = event;
    invalid.importance = f64::NAN;
    assert!(event_positions(&invalid, 10.0).is_err());
}

#[test]
fn exact_id_contract_rejects_unknown_missing_duplicate_and_invalid_values() {
    let directory = tempfile::tempdir().unwrap();
    let (candidates, ratings) = candidates(directory.path(), 3);
    validate_assessments(&candidates, &ratings).unwrap();
    assert!(validate_assessments(&candidates, &ratings[..2]).is_err());
    for invalid in ["unknown", "F0001"] {
        let mut changed = ratings.clone();
        changed[0].id = invalid.into();
        assert!(validate_assessments(&candidates, &changed).is_err());
    }
    for invalid in [f64::NAN, -1.0, 101.0, f64::INFINITY] {
        let mut changed = ratings.clone();
        changed[0].blog_score = invalid;
        assert!(validate_assessments(&candidates, &changed).is_err());
    }
    let mut changed = ratings;
    changed[0].scene = " \n ".into();
    assert!(validate_assessments(&candidates, &changed).is_err());
}

#[test]
fn tied_responses_and_candidate_permutations_have_the_same_order() {
    let directory = tempfile::tempdir().unwrap();
    let (mut candidates, mut ratings) = candidates(directory.path(), 3);
    for candidate in &mut candidates {
        candidate.quality.score = 80.0;
    }
    let expected = select(&candidates, &ratings, 3)
        .unwrap()
        .indices
        .iter()
        .map(|&index| candidates[index].id.clone())
        .collect::<Vec<_>>();
    assert_eq!(expected[0], "F0000");
    candidates.reverse();
    ratings.rotate_left(1);
    let actual = select(&candidates, &ratings, 3)
        .unwrap()
        .indices
        .iter()
        .map(|&index| candidates[index].id.clone())
        .collect::<Vec<_>>();
    assert_eq!(expected, actual);
}

#[test]
fn diversity_covers_sources_scenes_and_time_without_menu_dominance() {
    let directory = tempfile::tempdir().unwrap();
    let (mut candidates, mut ratings) = candidates(directory.path(), 8);
    for index in 0..8 {
        candidates[index].actual_seconds = (index % 3) as f64 * 10.0 + 1.0;
        candidates[index].requested_seconds = candidates[index].actual_seconds;
        candidates[index].source = if index < 4 { "A.mkv" } else { "B.mkv" }.into();
        ratings[index].scene = ["探索", "会話", "戦闘"][index % 3].into();
    }
    ratings[0].special = true;
    ratings[0].scene = "タイトル".into();
    ratings[0].blog_score = 100.0;
    ratings[1].special = true;
    ratings[1].scene = "地図".into();
    ratings[1].blog_score = 100.0;
    ratings[2].special = true;
    ratings[2].scene = "メニュー".into();
    ratings[2].blog_score = 100.0;
    let result = select(&candidates, &ratings, 4).unwrap();
    let chosen = &result.indices;
    assert!(!ratings[chosen[0]].special);
    assert_eq!(
        chosen
            .iter()
            .filter(|&&index| ratings[index].special)
            .count(),
        1
    );
    let sources: std::collections::BTreeSet<_> = chosen
        .iter()
        .map(|&index| &candidates[index].source)
        .collect();
    assert_eq!(sources.len(), 2);
    assert!(
        result
            .exclusions
            .iter()
            .any(|excluded| excluded.reason.contains("special screen"))
    );
    let result = select(&candidates, &ratings, 1).unwrap();
    assert!(!ratings[result.indices[0]].special);
    // Without special screens, valid slots cover all three time bands and Japanese scenes.
    for rating in &mut ratings {
        rating.special = false;
        rating.scene = "探索".into();
    }
    for candidate in &mut candidates {
        candidate.source = "A.mkv".into();
    }
    let result = select(&candidates, &ratings, 3).unwrap();
    let times: std::collections::BTreeSet<_> = result
        .indices
        .iter()
        .map(|&index| (candidates[index].actual_seconds / 10.0) as u8)
        .collect();
    assert_eq!(times.len(), 3);
}

#[test]
fn duplicates_transitions_and_quality_rejects_explain_shortage() {
    let directory = tempfile::tempdir().unwrap();
    let (mut candidates, mut ratings) = candidates(directory.path(), 5);
    candidates[1].quality = candidates[0].quality.clone();
    ratings[1].blog_score = 90.0;
    ratings[2].transition = true;
    candidates[3].quality.rejection = Some("dark/black frame".into());
    ratings[4].blog_score = 0.0;
    let error = select(&candidates, &ratings, 2).unwrap_err().to_string();
    assert!(error.contains("requested 2, available 1"));
    assert!(error.contains("duplicate") && error.contains("transition") && error.contains("dark"));
    let result = select_up_to(&candidates, &ratings, 5).unwrap();
    assert_eq!(result.indices, [1]);
}

#[test]
fn semantic_importance_selects_best_valid_event_and_keeps_its_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let (mut candidates, ratings) = candidates(directory.path(), 2);
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.event = Some(SemanticEvent {
            source: candidate.source.clone(),
            start_seconds: 0.0,
            end_seconds: 5.0,
            representative_seconds: candidate.actual_seconds,
            importance: if index == 0 { 10.0 } else { 100.0 },
            scene: "探索".into(),
            reason: format!("根拠 {index}"),
        });
    }
    let result = select(&candidates, &ratings, 1).unwrap();
    assert_eq!(result.indices, [1]);
    assert_eq!(candidates[1].event.as_ref().unwrap().reason, "根拠 1");
    candidates[1].event.as_mut().unwrap().source = "wrong.mkv".into();
    assert!(select(&candidates, &ratings, 1).is_err());
    candidates[1].event.as_mut().unwrap().source = candidates[1].source.clone();
    candidates[1].actual_seconds = 6.0;
    assert!(select(&candidates, &ratings, 1).is_err());
}

#[test]
fn normal_preference_reason_requires_an_actually_blocked_special_screen() {
    let directory = tempfile::tempdir().unwrap();
    let (candidates, mut ratings) = candidates(directory.path(), 4);
    for (normal_count, requested, expects_normal_preference) in
        [(0, 1, false), (1, 3, false), (1, 1, true)]
    {
        for (index, rating) in ratings.iter_mut().enumerate() {
            rating.special = index >= normal_count;
        }
        let result = select(&candidates, &ratings, requested).unwrap();
        assert_eq!(result.indices.len(), requested);
        assert!(result.exclusions.iter().all(|exclusion| {
            exclusion.reason.contains("favor normal progress") == expects_normal_preference
        }));
    }
}
