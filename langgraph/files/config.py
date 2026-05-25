REGISTRATION_CONFIGS = {
    1: {  # Attempt 1
        'name': 'Attempt 1',
        'affine': {
            'histogram_bins': 20,
            'sampling_percentage': 0.05,
            'learning_rate': 0.1,
            'max_iterations': 200,
            'convergence_value': 1e-8,
            'convergence_window': 2,
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 50,
            'sigma': 2.0,
            'shrink_factors': [2, 1],
            'smoothing_sigmas': [1, 0]
        }
    },
    2: {  # Attempt 2
        'name': 'Attempt 2',
        'affine': {
            'histogram_bins': 30,
            'sampling_percentage': 0.1,
            'learning_rate': 0.05,
            'max_iterations': 400,
            'convergence_value': 1e-9,
            'convergence_window': 5,
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 100,
            'sigma': 1.5,
            'shrink_factors': [4, 2, 1],
            'smoothing_sigmas': [2, 1, 0]
        }
    },
    3: {  # Attempt 3
        'name': 'Attempt 3',
        'affine': {
            'histogram_bins': 40,
            'sampling_percentage': 0.15,
            'learning_rate': 0.02,
            'max_iterations': 600,
            'convergence_value': 1e-10,
            'convergence_window': 10,
            'shrink_factors': [16, 8, 4, 2, 1],
            'smoothing_sigmas': [4, 3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 150,
            'sigma': 1.0,  # =============================
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3, 2, 1, 0]
        }
    },
    4: {  # Attempt 4
        'name': 'Attempt 4',
        'affine': {
            'histogram_bins': 50,
            'sampling_percentage': 0.2,
            'learning_rate': 0.01,
            'max_iterations': 350,
            'convergence_value': 1e-11,
            'convergence_window': 15,
            'shrink_factors': [16, 8, 4, 2, 1],
            'smoothing_sigmas': [4, 3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 200,
            'sigma': 0.5,
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3, 2, 1, 0]
        }
    },

    5: {  # Attempt 5
        'name': 'Attempt 5',
        'affine': {
            'histogram_bins': 60,
            'sampling_percentage': 0.3,
            'learning_rate': 0.005,
            'max_iterations': 400,
            'convergence_value': 1e-12,
            'convergence_window': 20,
            'shrink_factors': [16, 8, 4, 2, 1],
            'smoothing_sigmas': [4, 3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 300,
            'sigma': 0.3,
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3, 2, 1, 0]
        }
    },
}
