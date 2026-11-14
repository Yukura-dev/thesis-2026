import numpy as np
import pytest
import torch

import decoder_diagnostic as diagnostic
from training import Autoencoder


def test_sample_variation_is_not_spatial_or_globally_pooled_sd():
    # A fixed nonuniform image has spatial variance but no response to varying z.
    pattern = np.linspace(-2, 3, 2 * 64 * 64).reshape(2, 64, 64)
    repeated = np.repeat(pattern[None], 7, axis=0)
    measured = diagnostic.PixelPopulationMoments()
    measured.update(repeated[:3])
    measured.update(repeated[3:])
    assert measured.result() < 1e-15
    assert np.std(repeated) > 1


def test_population_denominator_and_chunk_invariance():
    # Population SD of {0, 2} is 1, whereas the sample SD would be sqrt(2).
    samples = np.broadcast_to(np.array([0., 2.])[:, None, None, None], (2, 2, 64, 64))
    measured = diagnostic.PixelPopulationMoments()
    measured.update(samples[:1])
    measured.update(samples[1:])
    assert measured.n == 2 and measured.result() == 1

    rng = np.random.default_rng(719)
    # Small changes about a large baseline exercise centered, float64 accumulation.
    samples = 1e6 + rng.normal(0, 1e-3, (11, 2, 64, 64))
    expected = np.std(samples, axis=0, ddof=0, dtype=np.float64).mean()
    for sizes in ((11,), (1,) * 11, (3, 5, 3)):
        moments = diagnostic.PixelPopulationMoments()
        offset = 0
        for size in sizes:
            moments.update(samples[offset:offset + size])
            offset += size
        assert moments.result() == pytest.approx(expected, abs=2e-11)


def test_parameterized_decoder_layers_include_fc_and_groupnorm_affine():
    model = Autoencoder(16, .3)
    rows = diagnostic.parameter_norms(model)
    assert list(rows) == ["from_latent", "decoder.0", "decoder.1", "decoder.3", "decoder.4",
                          "decoder.6", "decoder.7", "decoder.9"]
    assert rows["from_latent"]["layer_type"] == "Linear"
    assert rows["decoder.1"]["layer_type"] == "GroupNorm"
    assert rows["decoder.1"]["weight_kind"] == "affine scale vector"
    assert rows["decoder.1"]["weight_frobenius"] == pytest.approx(24**.5)
    assert rows["decoder.0"]["bias_l2"] is None
    assert rows["decoder.1"]["bias_l2"] == 0
    assert rows["decoder.9"]["bias_l2"] == 0
    expected = torch.linalg.vector_norm(model.decoder[9].weight.detach().double()).item()
    assert rows["decoder.9"]["weight_frobenius"] == pytest.approx(expected, abs=1e-15)


def test_missing_weights_never_reconstruct_or_train_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "CACHE", tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Do not initialize a replacement for missing saved weights")

    monkeypatch.setattr(diagnostic, "build_model", forbidden)
    with pytest.raises(FileNotFoundError):
        diagnostic.measure(None, {"config_hash": "missing"}, None, None)
