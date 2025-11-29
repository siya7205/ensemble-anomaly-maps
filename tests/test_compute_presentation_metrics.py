#!/usr/bin/env python3
"""
Tests for compute_presentation_metrics.py

Run with: pytest tests/test_compute_presentation_metrics.py -v
"""
import sys
import subprocess
from pathlib import Path
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_predictions_path():
    """Return path to sample predictions CSV."""
    return Path(__file__).parent / 'sample_predictions.csv'


@pytest.fixture
def output_dir(tmp_path):
    """Return temporary output directory."""
    return tmp_path / 'metrics_output'


class TestComputePresentationMetrics:
    """Test suite for compute_presentation_metrics.py"""
    
    def test_sample_predictions_exists(self, sample_predictions_path):
        """Verify sample predictions file exists."""
        assert sample_predictions_path.exists(), f"Sample predictions not found: {sample_predictions_path}"
    
    def test_sample_predictions_format(self, sample_predictions_path):
        """Verify sample predictions has correct format."""
        df = pd.read_csv(sample_predictions_path)
        
        # Check required columns
        assert 'frame' in df.columns, "Missing 'frame' column"
        assert 'y_true' in df.columns, "Missing 'y_true' column"
        assert 'y_score' in df.columns, "Missing 'y_score' column"
        
        # Check data types
        assert df['y_true'].isin([0, 1]).all(), "y_true should be binary (0 or 1)"
        assert (df['y_score'] >= 0).all() and (df['y_score'] <= 1).all(), "y_score should be in [0, 1]"
        
        # Check we have both classes
        assert df['y_true'].nunique() == 2, "Need both positive and negative samples"
    
    def test_script_runs_successfully(self, sample_predictions_path, output_dir):
        """Test that the script runs and produces expected outputs."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
        
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                '--predictions', str(sample_predictions_path),
                '--out-dir', str(output_dir),
                '--bootstrap', '200',  # Fewer resamples for fast CI
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        # Print output for debugging
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
    
    def test_output_files_created(self, sample_predictions_path, output_dir):
        """Test that expected output files are created."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
        
        subprocess.run(
            [
                sys.executable, str(script_path),
                '--predictions', str(sample_predictions_path),
                '--out-dir', str(output_dir),
                '--bootstrap', '200',
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        # Check that key output files exist
        assert (output_dir / 'metrics_summary.csv').exists(), "metrics_summary.csv not created"
        assert (output_dir / 'predictions_roc.png').exists(), "predictions_roc.png not created"
        assert (output_dir / 'predictions_pr.png').exists(), "predictions_pr.png not created"
        assert (output_dir / 'score_distributions.png').exists(), "score_distributions.png not created"
    
    def test_metrics_summary_content(self, sample_predictions_path, output_dir):
        """Test that metrics_summary.csv has expected content."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
        
        subprocess.run(
            [
                sys.executable, str(script_path),
                '--predictions', str(sample_predictions_path),
                '--out-dir', str(output_dir),
                '--bootstrap', '200',
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        # Load and verify metrics
        metrics_df = pd.read_csv(output_dir / 'metrics_summary.csv')
        
        # Check required columns
        required_cols = ['auroc', 'auroc_lo', 'auroc_hi', 'auprc', 'auprc_lo', 'auprc_hi']
        for col in required_cols:
            assert col in metrics_df.columns, f"Missing column: {col}"
        
        # Check values are in valid ranges
        assert 0 <= metrics_df['auroc'].iloc[0] <= 1, "AUROC should be in [0, 1]"
        assert 0 <= metrics_df['auprc'].iloc[0] <= 1, "AUPRC should be in [0, 1]"
        assert metrics_df['auroc_lo'].iloc[0] <= metrics_df['auroc'].iloc[0], "auroc_lo should be <= auroc"
        assert metrics_df['auroc_hi'].iloc[0] >= metrics_df['auroc'].iloc[0], "auroc_hi should be >= auroc"
    
    def test_per_run_metrics(self, sample_predictions_path, output_dir):
        """Test that per-run metrics are computed when run_id is present."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
        
        # Check if sample data has run_id
        df = pd.read_csv(sample_predictions_path)
        has_run_id = 'run_id' in df.columns and df['run_id'].nunique() > 1
        
        subprocess.run(
            [
                sys.executable, str(script_path),
                '--predictions', str(sample_predictions_path),
                '--out-dir', str(output_dir),
                '--bootstrap', '200',
                '--seed', '42'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        if has_run_id:
            assert (output_dir / 'metrics_summary_per_run.csv').exists(), "Per-run metrics not created"
            per_run_df = pd.read_csv(output_dir / 'metrics_summary_per_run.csv')
            assert 'run_id' in per_run_df.columns, "Missing run_id column in per-run metrics"
    
    def test_dry_run(self, sample_predictions_path):
        """Test dry run mode doesn't create output files."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
        
        result = subprocess.run(
            [
                sys.executable, str(script_path),
                '--predictions', str(sample_predictions_path),
                '--dry-run'
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        assert result.returncode == 0, f"Dry run failed: {result.stderr}"
        assert "DRY RUN" in result.stdout or "DRY RUN" in result.stderr, "Dry run message not found"
    
    def test_reproducibility(self, sample_predictions_path, tmp_path):
        """Test that results are reproducible with same seed."""
        script_path = Path(__file__).parent.parent / 'tools' / 'compute_presentation_metrics.py'
        
        out_dir1 = tmp_path / 'run1'
        out_dir2 = tmp_path / 'run2'
        
        # Run twice with same seed
        for out_dir in [out_dir1, out_dir2]:
            subprocess.run(
                [
                    sys.executable, str(script_path),
                    '--predictions', str(sample_predictions_path),
                    '--out-dir', str(out_dir),
                    '--bootstrap', '200',
                    '--seed', '42'
                ],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent)
            )
        
        # Compare metrics
        metrics1 = pd.read_csv(out_dir1 / 'metrics_summary.csv')
        metrics2 = pd.read_csv(out_dir2 / 'metrics_summary.csv')
        
        assert metrics1['auroc'].iloc[0] == metrics2['auroc'].iloc[0], "AUROC not reproducible"
        assert metrics1['auprc'].iloc[0] == metrics2['auprc'].iloc[0], "AUPRC not reproducible"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
