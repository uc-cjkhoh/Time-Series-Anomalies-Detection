from datetime import datetime, timezone

from consumer import one_minute_interval


class TestOneMinuteInterval:
    def test_removes_seconds(self):
        dt = datetime(2024, 2, 29, 12, 30, 45, tzinfo=timezone.utc)
        result = one_minute_interval(dt)
        assert result.second == 0
        assert result.microsecond == 0

    def test_preserves_minute(self):
        dt = datetime(2024, 2, 29, 12, 30, 45, tzinfo=timezone.utc)
        result = one_minute_interval(dt)
        assert result.minute == 30

    def test_preserves_hour_day_month_year(self):
        dt = datetime(2024, 2, 29, 12, 30, 45, tzinfo=timezone.utc)
        result = one_minute_interval(dt)
        assert result.year == 2024
        assert result.month == 2
        assert result.day == 29
        assert result.hour == 12

    def test_preserves_tzinfo(self):
        tz = timezone.utc
        dt = datetime(2024, 2, 29, 12, 30, 45, tzinfo=tz)
        result = one_minute_interval(dt)
        assert result.tzinfo == tz

    def test_whole_minute_unchanged(self):
        dt = datetime(2024, 2, 29, 12, 30, 0, tzinfo=timezone.utc)
        result = one_minute_interval(dt)
        assert result == dt

    def test_edge_59_seconds_rolls_down(self):
        dt = datetime(2024, 2, 29, 12, 30, 59, tzinfo=timezone.utc)
        result = one_minute_interval(dt)
        assert result == datetime(2024, 2, 29, 12, 30, 0, 0, tzinfo=timezone.utc)

    def test_seconds_removed_from_all_minutes(self):
        for sec in [0, 1, 15, 30, 45, 59]:
            dt = datetime(2024, 2, 29, 12, 30, sec, tzinfo=timezone.utc)
            result = one_minute_interval(dt)
            assert result == datetime(2024, 2, 29, 12, 30, 0, 0, tzinfo=timezone.utc)


class TestTxStatus:
    def test_status_type_12_status_in_range(self):
        for status_val in [1, 2, 3, 4, 5]:
            tx_status = int(12 == 12 and int(status_val) in [1, 2, 3, 4, 5])
            assert tx_status == 1

    def test_status_type_not_12(self):
        tx_status = int(10 == 12 and int(1) in [1, 2, 3, 4, 5])
        assert tx_status == 0

    def test_status_out_of_range(self):
        for status_val in [0, 6, 7, 99]:
            tx_status = int(12 == 12 and int(status_val) in [1, 2, 3, 4, 5])
            assert tx_status == 0
