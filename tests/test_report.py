from datetime import datetime, timezone

from src.classes.report import ReportSnapshot, PeriodicReport, FeaturesExtraction, AnomalyReport


class TestReportSnapshot:
    def test_report_id_format(self):
        snapshot = ReportSnapshot(
            mcc=505,
            mnc='12',
            rat=4,
            bound_type=1,
            dt=datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc),
            tx_succ_count=100,
            tx_total_count=150,
        )
        assert snapshot.report_id == '505-12-4-1'

    def test_report_id_varying_values(self):
        cases = [
            (460, '01', 4, 0, '460-01-4-0'),
            (505, '02', 7, 1, '505-02-7-1'),
            (310, '410', 5, 2, '310-410-5-2'),
        ]
        for mcc, mnc, rat, bound_type, expected in cases:
            snapshot = ReportSnapshot(
                mcc=mcc,
                mnc=mnc,
                rat=rat,
                bound_type=bound_type,
                dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
                tx_succ_count=0,
                tx_total_count=0,
            )
            assert snapshot.report_id == expected


class TestPeriodicReport:
    def test_initial_state(self):
        report = PeriodicReport(505, '12', 4, 1)
        assert report.mcc == 505
        assert report.mnc == '12'
        assert report.rat == 4
        assert report.bound_type == 1
        assert report.dt is None
        assert report.tx_succ_count == 0
        assert report.tx_total_count == 0

    def test_initial_state_with_args(self):
        dt = datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc)
        report = PeriodicReport(505, '12', 4, 1, dt=dt, tx_succ_count=10, tx_total_count=20)
        assert report.dt == dt
        assert report.tx_succ_count == 10
        assert report.tx_total_count == 20

    def test_record_tx_success(self):
        report = PeriodicReport(505, '12', 4, 1)
        report.record_tx(1)
        assert report.tx_total_count == 1
        assert report.tx_succ_count == 1

        report.record_tx(0)
        assert report.tx_total_count == 2
        assert report.tx_succ_count == 1

        report.record_tx(1)
        assert report.tx_total_count == 3
        assert report.tx_succ_count == 2

    def test_record_tx_many(self):
        report = PeriodicReport(505, '12', 4, 1)
        for _ in range(5):
            report.record_tx(1)
        for _ in range(3):
            report.record_tx(0)
        assert report.tx_total_count == 8
        assert report.tx_succ_count == 5

    def test_reset_tx(self):
        dt1 = datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2024, 2, 29, 12, 1, 0, tzinfo=timezone.utc)

        report = PeriodicReport(505, '12', 4, 1, dt=dt1, tx_succ_count=10, tx_total_count=20)
        report.reset_tx(dt2)
        assert report.dt == dt2
        assert report.tx_succ_count == 0
        assert report.tx_total_count == 0


class TestFeaturesExtraction:
    def test_update_and_extract_returns_dict(self):
        features = FeaturesExtraction(window_size=10)
        snapshot = ReportSnapshot(
            mcc=505, mnc='12', rat=4, bound_type=1,
            dt=datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc),
            tx_succ_count=100,
            tx_total_count=150,
        )
        result = features.update_and_extract(snapshot)
        assert isinstance(result, dict)
        assert 'dt' in result
        assert 'succ_tx_count' in result
        assert 'total_tx_count' in result
        assert 'succ_tx_ma' in result
        assert 'succ_tx_mvar' in result
        assert 'succ_tx_median' in result
        assert 'succ_count_z_score' in result
        assert 'total_tx_ma' in result
        assert 'total_tx_mvar' in result
        assert 'total_tx_median' in result
        assert 'total_count_z_score' in result

    def test_first_update_returns_self_as_ma(self):
        features = FeaturesExtraction(window_size=10)
        snapshot = ReportSnapshot(
            mcc=505, mnc='12', rat=4, bound_type=1,
            dt=datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc),
            tx_succ_count=100,
            tx_total_count=150,
        )
        result = features.update_and_extract(snapshot)
        assert result['succ_tx_count'] == 100
        assert result['total_tx_count'] == 150
        assert result['succ_tx_ma'] == 100
        assert result['total_tx_ma'] == 150

    def test_z_score_zero_on_first_point(self):
        features = FeaturesExtraction(window_size=10)
        snapshot = ReportSnapshot(
            mcc=505, mnc='12', rat=4, bound_type=1,
            dt=datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc),
            tx_succ_count=100,
            tx_total_count=150,
        )
        result = features.update_and_extract(snapshot)
        assert result['succ_count_z_score'] == 0.0
        assert result['total_count_z_score'] == 0.0

    def test_rolling_ma_converges(self):
        features = FeaturesExtraction(window_size=10)
        for i in range(10):
            snapshot = ReportSnapshot(
                mcc=505, mnc='12', rat=4, bound_type=1,
                dt=datetime(2024, 2, 29, 12, 0, i, tzinfo=timezone.utc),
                tx_succ_count=100,
                tx_total_count=150,
            )
            result = features.update_and_extract(snapshot)
            assert result['succ_tx_ma'] == 100
            assert result['total_tx_ma'] == 150

    def test_median_after_many_updates(self):
        features = FeaturesExtraction(window_size=20)
        for i in range(30):
            snapshot = ReportSnapshot(
                mcc=505, mnc='12', rat=4, bound_type=1,
                dt=datetime(2024, 2, 29, 12, 0, i, tzinfo=timezone.utc),
                tx_succ_count=100 + i,
                tx_total_count=150 + i,
            )
            features.update_and_extract(snapshot)
        median = features.succ_tx_median.get()
        assert median is not None
        assert median > 0


class TestAnomalyReport:
    def test_init_with_window_size(self):
        report = AnomalyReport(window_size=20)
        assert report.model is not None

    def test_update_model_accepts_dict(self):
        report = AnomalyReport(window_size=10)
        report.update_model({
            'dt': 1709208000000,
            'succ_tx_count': 100,
            'succ_tx_ma': 100.0,
            'succ_tx_mvar': 0.0,
            'succ_tx_median': 100.0,
            'succ_count_z_score': 0.0,
            'total_tx_count': 150,
            'total_tx_ma': 150.0,
            'total_tx_mvar': 0.0,
            'total_tx_median': 150.0,
            'total_count_z_score': 0.0,
        })

    def test_get_anomalies_score_returns_float(self):
        report = AnomalyReport(window_size=10)
        x = {
            'dt': 1709208000000,
            'succ_tx_count': 100,
            'succ_tx_ma': 100.0,
            'succ_tx_mvar': 0.0,
            'succ_tx_median': 100.0,
            'succ_count_z_score': 0.0,
            'total_tx_count': 150,
            'total_tx_ma': 150.0,
            'total_tx_mvar': 0.0,
            'total_tx_median': 150.0,
            'total_count_z_score': 0.0,
        }
        score = report.get_anomalies_score(x)
        assert isinstance(score, (int, float))

    def test_anomaly_score_responds_to_extreme_values(self):
        report = AnomalyReport(window_size=50)

        base = {
            'dt': 1709208000000,
            'succ_tx_ma': 100.0,
            'succ_tx_mvar': 10.0,
            'succ_tx_median': 100.0,
            'total_tx_ma': 150.0,
            'total_tx_mvar': 10.0,
            'total_tx_median': 150.0,
        }

        for i in range(60):
            x = {
                **base,
                'succ_tx_count': 100 + (i % 5),
                'succ_count_z_score': (i % 5 - 2) * 0.5,
                'total_tx_count': 150 + (i % 5),
                'total_count_z_score': (i % 5 - 2) * 0.5,
            }
            report.get_anomalies_score(x)

        normal = {**base, 'succ_tx_count': 100, 'succ_count_z_score': 0.0, 'total_tx_count': 150, 'total_count_z_score': 0.0}
        extreme = {**base, 'succ_tx_count': 500, 'succ_count_z_score': 10.0, 'total_tx_count': 600, 'total_count_z_score': 10.0}

        normal_score = report.get_anomalies_score(normal)
        extreme_score = report.get_anomalies_score(extreme)

        assert isinstance(normal_score, (int, float))
        assert isinstance(extreme_score, (int, float))
        assert extreme_score != normal_score
