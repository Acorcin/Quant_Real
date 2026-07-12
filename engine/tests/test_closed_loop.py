import unittest
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timezone
import json

from events.extractor import EventExtractor
from opportunity.measurement import OpportunityMeasurer
from learning.feedback import ClosedLoopLearner

class TestClosedLoopArchitecture(unittest.TestCase):
    
    @patch('events.extractor.get_connection')
    def test_event_extractor_active_signal(self, mock_get_conn):
        # Setup mocks
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchone.return_value = (42,) # Mock RETURNING id
        
        extractor = EventExtractor()
        
        composite = {
            "composite_direction": "long",
            "composite_strength": 0.85,
            "signal_count": 3,
            "tier2_count": 2,
            "eod_event_reversal": 1,
            "event_surprise_magnitude": 1.2
        }
        
        prediction = {
            "direction": "long",
            "probability": 0.76,
            "size_multiplier": 1.0,
            "model_version": "xgb_test_v1"
        }
        
        run_date = date(2026, 5, 24)
        result = extractor.extract_and_store(run_date, "EUR_USD", composite, prediction)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], 42)
        self.assertEqual(result["direction"], "long")
        self.assertEqual(result["magnitude"], 0.85)
        self.assertEqual(result["confidence"], 0.76)
        self.assertEqual(result["metadata"]["size_multiplier"], 1.0)
        
        # Verify SQL execution
        mock_cur.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch('events.extractor.get_connection')
    def test_event_extractor_flat_signal(self, mock_get_conn):
        extractor = EventExtractor()
        composite = {"composite_direction": "flat", "composite_strength": 0.0}
        prediction = {"direction": "flat", "probability": 0.50}
        
        result = extractor.extract_and_store(date(2026, 5, 24), "EUR_USD", composite, prediction)
        
        self.assertIsNone(result)
        mock_get_conn.assert_not_called()

    @patch('opportunity.measurement.fetch_all')
    @patch('opportunity.measurement.get_connection')
    def test_opportunity_measurer_h4_bars(self, mock_get_conn, mock_fetch_all):
        # Mock H4 bars (6 bars for the day)
        mock_fetch_all.return_value = [
            {"bar_time": datetime(2026, 5, 24, 18, 0), "open": 1.0800, "high": 1.0820, "low": 1.0790, "close": 1.0810},
            {"bar_time": datetime(2026, 5, 24, 22, 0), "open": 1.0810, "high": 1.0850, "low": 1.0805, "close": 1.0840},
            {"bar_time": datetime(2026, 5, 25, 2, 0), "open": 1.0840, "high": 1.0860, "low": 1.0830, "close": 1.0850},
            {"bar_time": datetime(2026, 5, 25, 6, 0), "open": 1.0850, "high": 1.0880, "low": 1.0840, "close": 1.0870},
            {"bar_time": datetime(2026, 5, 25, 10, 0), "open": 1.0870, "high": 1.0890, "low": 1.0860, "close": 1.0865},
            {"bar_time": datetime(2026, 5, 25, 14, 0), "open": 1.0865, "high": 1.0870, "low": 1.0820, "close": 1.0830},
        ]
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        
        measurer = OpportunityMeasurer()
        
        # Test long opportunity
        result = measurer.measure_and_store(
            event_id=10,
            event_date=date(2026, 5, 24),
            trade_date=date(2026, 5, 25),
            instrument="EUR_USD",
            direction="long"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["entry_price"], 1.0800) # Open of first H4
        self.assertEqual(result["exit_price"], 1.0830)  # Close of last H4
        
        # Max high = 1.0890, Max low = 1.0790
        # MFE = 1.0890 - 1.0800 = 0.0090 (90 pips)
        # MAE = 1.0790 - 1.0800 = -0.0010 (-10 pips)
        # Close = 1.0830 - 1.0800 = 0.0030 (30 pips)
        self.assertAlmostEqual(result["mfe_pips"], 90.0)
        self.assertAlmostEqual(result["mae_pips"], -10.0)
        self.assertAlmostEqual(result["close_return_pips"], 30.0)
        self.assertAlmostEqual(result["path_ratio"], 90.0 / (90.0 + 10.0))
        
        mock_conn.commit.assert_called_once()

    @patch('opportunity.measurement.fetch_one')
    @patch('opportunity.measurement.fetch_all')
    @patch('opportunity.measurement.get_connection')
    def test_opportunity_measurer_daily_fallback(self, mock_get_conn, mock_fetch_all, mock_fetch_one):
        # Mock H4 empty list, forcing Daily bar fallback
        mock_fetch_all.return_value = []
        mock_fetch_one.return_value = {
            "open": 1.0800,
            "high": 1.0870,
            "low": 1.0770,
            "close": 1.0820
        }
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        
        measurer = OpportunityMeasurer()
        
        # Test short opportunity
        result = measurer.measure_and_store(
            event_id=11,
            event_date=date(2026, 5, 24),
            trade_date=date(2026, 5, 25),
            instrument="EUR_USD",
            direction="short"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result["entry_price"], 1.0800)
        self.assertEqual(result["exit_price"], 1.0820)
        
        # For short:
        # MFE = 1.0800 - 1.0770 (low) = 30 pips
        # MAE = 1.0800 - 1.0870 (high) = -70 pips
        # Close = 1.0800 - 1.0820 = -20 pips
        self.assertAlmostEqual(result["mfe_pips"], 30.0)
        self.assertAlmostEqual(result["mae_pips"], -70.0)
        self.assertAlmostEqual(result["close_return_pips"], -20.0)
        self.assertAlmostEqual(result["path_ratio"], 30.0 / (30.0 + 70.0))
        
        mock_conn.commit.assert_called_once()

    @patch('learning.feedback.redis.Redis')
    @patch('learning.feedback.get_connection')
    @patch('learning.feedback.fetch_all')
    def test_closed_loop_learner_adaptation(self, mock_fetch_all, mock_get_conn, mock_redis):
        # Mock Redis client
        mock_r = MagicMock()
        mock_redis.return_value = mock_r
        
        # Mock DB connection
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_get_conn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        
        # 1. Simulate Overconfident model (calibration drift = 0.70 confidence - 0.40 win rate = 0.30)
        # We return 10 events: 4 wins (positive return), 6 losses (negative return)
        # Expected average confidence = 0.70
        opp_data = [
            {"confidence": 0.70, "sig_dir": "long", "close_return_pips": 20.0},
            {"confidence": 0.70, "sig_dir": "long", "close_return_pips": 15.0},
            {"confidence": 0.70, "sig_dir": "long", "close_return_pips": -10.0},
            {"confidence": 0.70, "sig_dir": "long", "close_return_pips": -25.0},
            {"confidence": 0.70, "sig_dir": "long", "close_return_pips": 5.0},
            {"confidence": 0.70, "sig_dir": "long", "close_return_pips": -30.0},
            {"confidence": 0.70, "sig_dir": "long", "close_return_pips": -5.0},
            {"confidence": 0.70, "sig_dir": "long", "close_return_pips": 12.0},
            {"confidence": 0.70, "sig_dir": "long", "close_return_pips": -18.0},
            {"confidence": 0.70, "sig_dir": "long", "close_return_pips": -8.0},
        ]
        
        # Mock slippage (let's say 2.5 pips of average entry slippage)
        slippage_data = [
            {"entry_diff": 0.00025, "exit_diff": 0.00010},
            {"entry_diff": 0.00025, "exit_diff": 0.00010},
        ]
        
        # Mock queries inside run_feedback_cycle:
        # First call fetches opportunity measurements.
        # Second call fetches live trades.
        # Third call fetches slippage calculations.
        mock_fetch_all.side_effect = [opp_data, [{"ticket_id": "test_ticket"}], slippage_data]
        
        learner = ClosedLoopLearner(lookback_days=20)
        result = learner.run_feedback_cycle(date(2026, 5, 24), "EUR_USD")
        
        self.assertIsNotNone(result)
        metrics = result["metrics"]
        params = result["adjusted_parameters"]
        
        # Drift = 0.70 - 0.40 = 0.30
        self.assertEqual(metrics["calibration_drift"], 0.30)
        self.assertEqual(metrics["avg_entry_slippage_pips"], 2.5)
        
        # Sizing / probability overrides should kick in due to drift > 0.05
        # kelly_fraction = 0.15 - 0.5 * (0.30 - 0.05) = 0.025 -> capped at min 0.05
        self.assertEqual(params["kelly_fraction"], 0.05)
        
        # Gates: half = 0.55 + 0.5 * (0.30 - 0.05) = 0.675 -> capped at max 0.65
        self.assertEqual(params["probability_threshold_half"], 0.65)
        
        # Slippage: avg_entry_slip = 2.5 pips. max_spread should tighten:
        # max_spread_pips = 2.0 - 0.5 * (2.5 - 1.0) = 1.25 pips
        self.assertEqual(params["max_spread_pips"], 1.25)
        
        # Velocity entry threshold should scale up from 1.5e-6:
        # velocity_entry_threshold = 1.5e-6 * (1.0 + 0.25 * (2.5 - 1.0)) = 1.5e-6 * 1.375 = 2.0625e-6
        self.assertGreater(params["velocity_entry_threshold"], 1.5e-6)
        
        # Verify Redis push was triggered with correct parameters
        mock_r.set.assert_called_with("EUR_USD:learning_params", json.dumps(params))
        mock_conn.commit.assert_called_once()

if __name__ == '__main__':
    unittest.main()
