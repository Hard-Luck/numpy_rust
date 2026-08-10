use numpy_rust::{Pcg64, target_numpy_version};

include!("fixtures/generated_fixtures.rs");

#[test]
fn doubles_match_numpy() {
    assert_eq!(target_numpy_version(), TARGET_NUMPY_VERSION);
    for fixture in FIXTURES {
        let mut rng = Pcg64::from_seed(fixture.seed);
        for (index, expected_bits) in fixture.doubles.iter().copied().enumerate() {
            assert_eq!(
                rng.next_double().to_bits(),
                expected_bits,
                "seed={} double_index={}",
                fixture.seed,
                index
            );
        }
    }
}

#[test]
fn normals_match_numpy() {
    for fixture in FIXTURES {
        let mut rng = Pcg64::from_seed(fixture.seed);
        for (index, expected_bits) in fixture.normals.iter().copied().enumerate() {
            assert_eq!(
                rng.standard_normal().to_bits(),
                expected_bits,
                "seed={} normal_index={}",
                fixture.seed,
                index
            );
        }
    }
}

#[test]
fn poisson_matches_numpy() {
    for fixture in FIXTURES {
        for (lambda_index, lambda) in POISSON_LAMBDAS.iter().copied().enumerate() {
            let mut rng = Pcg64::from_seed(fixture.seed);
            for (index, expected) in fixture.poisson[lambda_index].iter().copied().enumerate() {
                assert_eq!(
                    rng.poisson(lambda),
                    expected,
                    "seed={} lambda={} sample_index={}",
                    fixture.seed,
                    lambda,
                    index
                );
            }
        }
    }
}

#[test]
fn poisson_array_matches_repeated_scalar_draws() {
    for fixture in FIXTURES {
        for lambda in POISSON_LAMBDAS {
            let mut array_rng = Pcg64::from_seed(fixture.seed);
            let mut scalar_rng = Pcg64::from_seed(fixture.seed);

            let array_samples = array_rng.poisson_array(lambda, 53);
            let scalar_samples = (0..53)
                .map(|_| scalar_rng.poisson(lambda))
                .collect::<Vec<_>>();

            assert_eq!(
                array_samples, scalar_samples,
                "seed={} lambda={}",
                fixture.seed, lambda
            );
        }
    }
}

#[test]
fn shared_rng_bulk_draws_match_scalar_draw_order() {
    for fixture in FIXTURES {
        let lambda = 2.0;
        let poisson_count = 53;
        let normal_count = 17;
        let loc = 0.0;
        let scale = 1.75;

        let mut bulk_rng = Pcg64::from_seed(fixture.seed);
        let bulk_events = bulk_rng.poisson_array(lambda, poisson_count);
        let bulk_normals = bulk_rng.normal(loc, scale, normal_count);

        let mut scalar_rng = Pcg64::from_seed(fixture.seed);
        let scalar_events = (0..poisson_count)
            .map(|_| scalar_rng.poisson(lambda))
            .collect::<Vec<_>>();
        let scalar_normals = (0..normal_count)
            .map(|_| scalar_rng.normal_scalar(loc, scale))
            .collect::<Vec<_>>();

        assert_eq!(bulk_events, scalar_events, "seed={} poisson", fixture.seed);
        assert_eq!(
            bulk_normals
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            scalar_normals
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "seed={} normal",
            fixture.seed
        );
    }
}

#[test]
fn appliance_flow_matches_numpy_with_sequence_seed() {
    let seed = 101_u64;
    let flat_profile = [0.05, 0.1, 0.2, 0.25, 0.4];
    let annual_expected_uses: f64 = 730.0;
    let duration_std_dev: f64 = 2.0;

    let seed_words = (0..(flat_profile.len() + annual_expected_uses.ceil() as usize))
        .map(|index| seed + index as u64)
        .collect::<Vec<_>>();
    let lambdas = flat_profile
        .iter()
        .map(|value| value * annual_expected_uses / 365.0)
        .collect::<Vec<_>>();

    let mut appliance_rng = Pcg64::from_seed_slice(&seed_words);
    let events = appliance_rng.poisson_array_from_slice(&lambdas, lambdas.len());
    assert_eq!(events, vec![0, 0, 1, 0, 0]);

    let num_events = events
        .iter()
        .copied()
        .map(|value| value as usize)
        .sum::<usize>();
    let mut event_size_deviations = appliance_rng.normal(0.0, duration_std_dev, num_events);
    for deviation in &mut event_size_deviations {
        if *deviation < -1.0 {
            *deviation = appliance_rng.normal_scalar(0.0, duration_std_dev).max(-1.0);
        }
    }

    assert_eq!(
        event_size_deviations
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![0xbfb2ceab86095daa],
    );

    let norm_events = num_events as f64 + event_size_deviations.iter().sum::<f64>();
    assert_eq!(norm_events.to_bits(), 0x3feda62a8f3ed44b);
}

#[test]
fn random_examples_match_numpy() {
    for example in RANDOM_EXAMPLES {
        let mut doubles_rng = Pcg64::from_seed(example.seed);
        for (index, expected_bits) in example.doubles.iter().copied().enumerate() {
            assert_eq!(
                doubles_rng.next_double().to_bits(),
                expected_bits,
                "random_example seed={} double_index={}",
                example.seed,
                index
            );
        }

        let mut normals_rng = Pcg64::from_seed(example.seed);
        for (index, expected_bits) in example.normals.iter().copied().enumerate() {
            assert_eq!(
                normals_rng.standard_normal().to_bits(),
                expected_bits,
                "random_example seed={} normal_index={}",
                example.seed,
                index
            );
        }

        let mut poisson_rng = Pcg64::from_seed(example.seed);
        for (index, expected) in example.poisson.iter().copied().enumerate() {
            assert_eq!(
                poisson_rng.poisson(example.lambda),
                expected,
                "random_example seed={} lambda={} poisson_index={}",
                example.seed,
                example.lambda,
                index
            );
        }
    }
}
