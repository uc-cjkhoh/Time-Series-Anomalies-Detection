import os
import pickle
import tempfile
from unittest.mock import patch

from src.online_ml.architecture import _checkpoint_path, _save_state, _load_state
from src.classes.report import FeaturesExtraction, AnomalyReport


class TestCheckpointPath:
    def test_checkpoint_path_format(self):
        path = _checkpoint_path("505-12-4-1", "features")
        assert path == "./model_checkpoints/505-12-4-1__features.pkl"

    def test_checkpoint_path_anomaly(self):
        path = _checkpoint_path("505-12-4-1", "anomaly")
        assert path == "./model_checkpoints/505-12-4-1__anomaly.pkl"

    def test_checkpoint_path_different_id(self):
        path = _checkpoint_path("460-01-5-0", "features")
        assert path == "./model_checkpoints/460-01-5-0__features.pkl"


class TestSaveLoadState:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.online_ml.architecture.CHECKPOINT_DIR", tmpdir):
                features = FeaturesExtraction(window_size=10)
                anomaly = AnomalyReport(window_size=10)

                _save_state("test-group", features, anomaly)

                loaded_features, loaded_anomaly = _load_state("test-group", window_size=10)

                assert loaded_features is not None
                assert loaded_anomaly is not None

                feat_path = os.path.join(tmpdir, "test-group__features.pkl")
                anom_path = os.path.join(tmpdir, "test-group__anomaly.pkl")
                assert os.path.exists(feat_path)
                assert os.path.exists(anom_path)

    def test_load_state_when_no_checkpoint_returns_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.online_ml.architecture.CHECKPOINT_DIR", tmpdir):
                features, anomaly = _load_state("nonexistent", window_size=20)

                assert isinstance(features, FeaturesExtraction)
                assert isinstance(anomaly, AnomalyReport)

    def test_load_state_partial_checkpoint_returns_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.online_ml.architecture.CHECKPOINT_DIR", tmpdir):
                features = FeaturesExtraction(window_size=5)
                # Only save features, not anomaly
                os.makedirs(tmpdir, exist_ok=True)
                with open(os.path.join(tmpdir, "partial__features.pkl"), "wb") as f:
                    pickle.dump(features, f)

                loaded_features, loaded_anomaly = _load_state("partial", window_size=5)

                assert isinstance(loaded_features, FeaturesExtraction)
                assert isinstance(loaded_anomaly, AnomalyReport)

    def test_save_state_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "path")
            with patch("src.online_ml.architecture.CHECKPOINT_DIR", nested):
                features = FeaturesExtraction(window_size=5)
                anomaly = AnomalyReport(window_size=5)
                _save_state("g", features, anomaly)
                assert os.path.isdir(nested)

    def test_multiple_saves_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.online_ml.architecture.CHECKPOINT_DIR", tmpdir):
                feats1 = FeaturesExtraction(window_size=5)
                anom1 = AnomalyReport(window_size=5)

                feats2 = FeaturesExtraction(window_size=10)
                anom2 = AnomalyReport(window_size=10)

                _save_state("g", feats1, anom1)
                _save_state("g", feats2, anom2)

                loaded_features, _ = _load_state("g", window_size=5)
                assert loaded_features is not None
