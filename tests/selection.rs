use game_screen_pick::{
    quality::analyze,
    selection::{
        Assessment, Candidate, MAX_CANDIDATES, SemanticEvent, event_positions, sample_positions,
        select, select_up_to, validate_assessments,
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
fn nontransitive_duplicate_group_keeps_a_feasible_pair() {
    let directory = tempfile::tempdir().unwrap();
    let (mut frames, mut ratings) = candidates(directory.path(), 1);
    let original = image::open(&frames[0].image).unwrap().to_rgb8();
    ratings[0].blog_score = 100.0;
    for (index, delta) in [(1, -6_i16), (2, 6_i16)] {
        let mut image = original.clone();
        for pixel in image.pixels_mut() {
            for value in &mut pixel.0 {
                *value = (i16::from(*value) + delta).clamp(0, 255) as u8;
            }
        }
        let path = directory.path().join(format!("exposure-{index}.png"));
        image.save(&path).unwrap();
        let mut frame = frames[0].clone();
        frame.id = format!("F{index:04}");
        frame.image = path;
        frame.quality = analyze(&frame.image).unwrap();
        frame.requested_seconds = index as f64;
        frame.actual_seconds = index as f64;
        let mut rating = ratings[0].clone();
        rating.id = frame.id.clone();
        rating.blog_score = 90.0;
        frames.push(frame);
        ratings.push(rating);
    }
    assert!(frames.iter().all(|frame| frame.quality.rejection.is_none()));
    assert!(frames[0].quality.near_duplicate(&frames[1].quality));
    assert!(frames[0].quality.near_duplicate(&frames[2].quality));
    assert!(!frames[1].quality.near_duplicate(&frames[2].quality));
    let selected = select(&frames, &ratings, 2).unwrap();
    let ids = selected
        .indices
        .iter()
        .map(|&index| frames[index].id.clone())
        .collect::<Vec<_>>();
    assert_eq!(ids, ["F0001", "F0002"]);
    assert!(
        selected
            .exclusions
            .iter()
            .any(|item| item.id == "F0000" && item.reason.contains("near duplicate"))
    );
    frames.reverse();
    ratings.rotate_left(1);
    let permuted = select(&frames, &ratings, 2).unwrap();
    assert_eq!(
        permuted
            .indices
            .iter()
            .map(|&index| frames[index].id.clone())
            .collect::<Vec<_>>(),
        ids
    );
    for rating in &mut ratings {
        rating.special = true;
    }
    assert_eq!(select(&frames, &ratings, 2).unwrap().indices.len(), 2);
}

#[test]
fn duplicate_reselection_can_replace_two_representatives_and_preserves_normal_priority() {
    let directory = tempfile::tempdir().unwrap();
    let (template, _) = candidates(directory.path(), 1);
    let original = RgbImage::from_fn(96, 64, |x, y| {
        let tile = (x / 8 + y / 8 * 12) as usize;
        Rgb([
            32 + ((tile * 73) % 192) as u8,
            32 + ((tile * 41) % 192) as u8,
            32 + ((tile * 97) % 192) as u8,
        ])
    });
    let mut frames = Vec::new();
    let mut ratings = Vec::new();
    for (index, (id, red, green, score)) in [
        ("A", -24, 0, 100.0),
        ("D", 24, 0, 100.0),
        ("B", -30, -18, 90.0),
        ("C", 30, -18, 90.0),
        ("E", 0, 0, 90.0),
    ]
    .into_iter()
    .enumerate()
    {
        let mut image = original.clone();
        for pixel in image.pixels_mut() {
            pixel[0] = (i16::from(pixel[0]) + red) as u8;
            pixel[1] = (i16::from(pixel[1]) + green) as u8;
        }
        let path = directory.path().join(format!("{id}.png"));
        image.save(&path).unwrap();
        let mut frame = template[0].clone();
        frame.id = id.into();
        frame.image = path;
        frame.quality = analyze(&frame.image).unwrap();
        frame.actual_seconds = index as f64;
        frame.requested_seconds = index as f64;
        frames.push(frame);
        ratings.push(Assessment {
            id: id.into(),
            blog_score: score,
            transition: false,
            scene: "探索".into(),
            reason: "正常進行の実画像".into(),
            special: false,
        });
    }
    assert!(frames.iter().all(|frame| frame.quality.rejection.is_none()));
    for (left, right) in [(0, 2), (0, 4), (1, 3), (1, 4)] {
        assert!(frames[left].quality.near_duplicate(&frames[right].quality));
    }
    for (left, right) in [(0, 1), (2, 3), (2, 4), (3, 4)] {
        assert!(!frames[left].quality.near_duplicate(&frames[right].quality));
    }
    // A/D are the preferred hubs; B/C/E are the only feasible three-image set.
    for count in [3, 9] {
        let result = select_up_to(&frames, &ratings, count).unwrap();
        let chosen: std::collections::BTreeSet<_> = result
            .indices
            .iter()
            .map(|&index| frames[index].id.as_str())
            .collect();
        assert_eq!(chosen, std::collections::BTreeSet::from(["B", "C", "E"]));
        assert!(
            result
                .notes
                .iter()
                .any(|note| note.starts_with("3 normal progress / 0 special"))
        );
    }
    for rating in &mut ratings[2..] {
        rating.special = true;
    }
    let result = select_up_to(&frames, &ratings, 3).unwrap();
    assert_eq!(result.indices.len(), 2);
    assert!(result.indices.iter().all(|&index| !ratings[index].special));
}

#[test]
fn maximal_duplicate_pool_stops_without_filling_slots_with_repeats() {
    let directory = tempfile::tempdir().unwrap();
    let (template, ratings) = candidates(directory.path(), 1);
    let frames = (0..MAX_CANDIDATES)
        .map(|index| {
            let mut frame = template[0].clone();
            frame.id = format!("F{index:04}");
            frame
        })
        .collect::<Vec<_>>();
    let ratings = frames
        .iter()
        .map(|frame| {
            let mut rating = ratings[0].clone();
            rating.id = frame.id.clone();
            rating
        })
        .collect::<Vec<_>>();
    let result = select_up_to(&frames, &ratings, 999).unwrap();
    assert_eq!(result.indices, [0]);
    assert_eq!(result.exclusions.len(), MAX_CANDIDATES - 1);
    assert!(
        result
            .exclusions
            .iter()
            .all(|item| item.reason == "near duplicate of selected image")
    );
    assert!(
        result
            .notes
            .iter()
            .any(|note| note.contains("bounded alternatives"))
    );
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
    assert!(error.contains("requested 2, found 1"));
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
