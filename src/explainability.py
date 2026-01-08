"""
SHAP explainability module for ADR predictions
"""
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys
from typing import List, Tuple, Dict
import warnings
import xgboost as xgb
import json
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_model, load_metadata, get_data_path


class SHAPExplainer:
    """
    SHAP-based explainer for XGBoost ADR model
    """
    
    def __init__(self, model=None, model_path: str = "models/xgb_adr_model.json"):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model (optional)
            model_path: Path to saved model
        """
        if model is None:
            self.model = load_model(model_path)
        else:
            self.model = model
            
        self.explainer = None
        self.feature_names = []
        
        # Load feature manifest (priority)
        try:
            manifest_path = get_data_path("feature_manifest.json")
            with open(manifest_path, "r") as f:
                data = json.load(f)
                self.feature_names = data.get("features", [])
        except:
            # Fallback to metadata pkl
            try:
                metadata_path = model_path.replace(".json", "_metadata.pkl").replace(".pkl", "_metadata.pkl")
                if os.path.exists(metadata_path):
                    metadata = load_metadata(metadata_path)
                    self.feature_names = metadata.get('feature_names', [])
            except:
                print("Warning: Could not load feature names")
    
    def create_explainer(self, X_background: pd.DataFrame = None):
        """
        Create SHAP TreeExplainer
        
        Args:
            X_background: Background data for explainer (optional)
        """
        print("Creating SHAP TreeExplainer...")
        
        if X_background is not None:
            # Use subset of data as background (for speed)
            if len(X_background) > 100:
                background = shap.sample(X_background, 100)
            else:
                background = X_background
            self.explainer = shap.TreeExplainer(self.model, background)
        else:
            self.explainer = shap.TreeExplainer(self.model)
        
        print("✓ SHAP explainer created")
        
    def save_explainer(self, path: str = "models/shap_explainer.pkl"):
        """
        Save SHAP explainer
        
        Args:
            path: Path to save explainer
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.explainer, path)
        print(f"✓ SHAP explainer saved to {path}")
    
    def load_explainer(self, path: str = "models/shap_explainer.pkl"):
        """
        Load saved SHAP explainer
        
        Args:
            path: Path to explainer file
        """
        if os.path.exists(path):
            self.explainer = joblib.load(path)
            print(f"✓ SHAP explainer loaded from {path}")
        else:
            print(f"Warning: Explainer not found at {path}")
    
    def compute_shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """
        Compute SHAP values for given data
        
        Args:
            X: Input features (may include leakage features that will be dropped)
            
        Returns:
            SHAP Explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call create_explainer() first.")
        
        print(f"Computing SHAP values for {len(X)} samples...")
        
        # The model was trained WITHOUT leakage features, so we need to drop them
        # before passing to SHAP (same as we do for predictions)
        leakage = ['weak_score', 'high_risk_drug', 'faers_adr_rate', 'faers_severe_rate']
        X_clean = X.drop(columns=[c for c in leakage if c in X.columns], errors='ignore')
        
        # Ensure feature alignment with model
        # Get expected feature count from model
        try:
            # Try to get feature count from model
            if hasattr(self.model, 'num_features'):
                expected_feature_count = self.model.num_features()
            elif hasattr(self.model, 'get_score'):
                # Get feature count from importance scores
                importance = self.model.get_score(importance_type='gain')
                expected_feature_count = len(importance)
            else:
                expected_feature_count = None
        except Exception:
            expected_feature_count = None
        
        # Check if feature count matches after dropping leakage
        if expected_feature_count is not None and len(X_clean.columns) != expected_feature_count:
            print(f"Warning: Feature count mismatch. Expected: {expected_feature_count}, Got: {len(X_clean.columns)}")
            
            # Try to use feature_names if available (these should be the clean features)
            if self.feature_names and len(self.feature_names) == expected_feature_count:
                # Create aligned DataFrame using feature_names
                X_aligned = pd.DataFrame(index=X_clean.index)
                for feat in self.feature_names:
                    if feat in X_clean.columns:
                        X_aligned[feat] = X_clean[feat]
                    else:
                        # Add missing feature with zeros
                        X_aligned[feat] = 0
                        print(f"  Adding missing feature '{feat}' with zeros")
                X_clean = X_aligned
            else:
                # If we still have a mismatch, try to infer from model importance
                try:
                    importance = self.model.get_score(importance_type='gain')
                    # XGBoost uses f0, f1, f2... format, map to column indices
                    if importance and list(importance.keys())[0].startswith('f'):
                        # Model uses positional features, just ensure count matches
                        if len(X_clean.columns) > expected_feature_count:
                            # Drop extra columns (keep first N)
                            X_clean = X_clean.iloc[:, :expected_feature_count]
                        elif len(X_clean.columns) < expected_feature_count:
                            # Add missing columns with zeros
                            missing = expected_feature_count - len(X_clean.columns)
                            for i in range(missing):
                                X_clean[f'missing_feature_{i}'] = 0
                except Exception:
                    pass
                
                # Final check
                if len(X_clean.columns) != expected_feature_count:
                    raise ValueError(
                        f"Feature count mismatch: Model expects {expected_feature_count} features, "
                        f"but received {len(X_clean.columns)} after dropping leakage. "
                        f"Original had {len(X.columns)} features. "
                        f"Please ensure X contains all features the model was trained on (excluding leakage)."
                    )
        
        # Use the cleaned DataFrame
        X = X_clean
        
        try:
            shap_values = self.explainer(X)
            print("✓ SHAP values computed")
        except Exception as e:
            # If still fails, provide more detailed error
            error_msg = f"Error computing SHAP: {e}\n"
            error_msg += f"X shape: {X.shape}, columns: {len(X.columns)}\n"
            if expected_feature_count:
                error_msg += f"Expected feature count: {expected_feature_count}\n"
            if self.feature_names:
                error_msg += f"Feature names available: {len(self.feature_names)}\n"
            print(error_msg)
            raise ValueError(error_msg) from e
        
        return shap_values
    
    def get_global_importance(
        self, 
        X: pd.DataFrame, 
        top_n: int = 20,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Get global feature importance using SHAP
        
        Args:
            X: Input features
            top_n: Number of top features to return
            save_path: Optional path to save plot
            
        Returns:
            DataFrame with feature importances
        """
        print("\nComputing global feature importance...")
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X)
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        
        # Create DataFrame
        # X has been cleaned (leakage removed) in compute_shap_values
        leakage = ['weak_score', 'high_risk_drug', 'faers_adr_rate', 'faers_severe_rate']
        X_clean = X.drop(columns=[c for c in leakage if c in X.columns], errors='ignore')
        
        if self.feature_names and len(self.feature_names) == len(mean_abs_shap):
            feature_names = self.feature_names
        elif len(X_clean.columns) == len(mean_abs_shap):
            feature_names = X_clean.columns.tolist()
        else:
            # Fallback to generic names
            feature_names = [f'feature_{i}' for i in range(len(mean_abs_shap))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        # Print top features
        print(f"\nTop {top_n} features by SHAP importance:")
        for i, row in importance_df.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Create plot if requested
        if save_path:
            self.plot_global_importance(shap_values, save_path)
        
        return importance_df
    
    def plot_global_importance(
        self, 
        shap_values: shap.Explanation,
        save_path: str = "reports/shap_global_importance.png",
        max_display: int = 20
    ):
        """
        Plot global SHAP summary
        
        Args:
            shap_values: SHAP values
            save_path: Path to save plot
            max_display: Maximum features to display
        """
        print(f"\nCreating global SHAP summary plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values.values, 
            shap_values.data,
            feature_names=self.feature_names if self.feature_names else None,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Global importance plot saved to {save_path}")
    
    def get_local_explanation(
        self, 
        X_single: pd.DataFrame,
        top_n: int = 10
    ) -> Tuple[List[Tuple[str, float]], shap.Explanation]:
        """
        Get local explanation for single prediction
        
        Args:
            X_single: Single patient features (1 row DataFrame) - may include leakage features
            top_n: Number of top contributors to return
            
        Returns:
            Tuple of (top_contributors list, shap_values)
        """
        if len(X_single) != 1:
            raise ValueError("X_single must contain exactly 1 sample")
        
        # Compute SHAP values (this will drop leakage features internally)
        shap_values = self.compute_shap_values(X_single)
        
        # Get feature names - use the cleaned features (after leakage removal)
        # The compute_shap_values method returns values for cleaned features
        leakage = ['weak_score', 'high_risk_drug', 'faers_adr_rate', 'faers_severe_rate']
        X_clean = X_single.drop(columns=[c for c in leakage if c in X_single.columns], errors='ignore')
        
        if self.feature_names and len(self.feature_names) == len(shap_values.values[0]):
            feature_names = self.feature_names
        else:
            # Use column names from cleaned DataFrame
            feature_names = X_clean.columns.tolist()
            # If still mismatch, use indices
            if len(feature_names) != len(shap_values.values[0]):
                feature_names = [f'feature_{i}' for i in range(len(shap_values.values[0]))]
        
        # Create list of (feature, shap_value) tuples
        shap_contributions = list(zip(feature_names, shap_values.values[0]))
        
        # Sort by absolute contribution
        shap_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top N
        top_contributors = shap_contributions[:top_n]
        
        return top_contributors, shap_values
    
    def plot_local_explanation(
        self,
        shap_values: shap.Explanation,
        save_path: str = None,
        plot_type: str = "waterfall"
    ) -> plt.Figure:
        """
        Plot local explanation for single prediction
        
        Args:
            shap_values: SHAP values for single sample
            save_path: Optional path to save plot
            plot_type: Type of plot ('waterfall' or 'force')
            
        Returns:
            Matplotlib figure
        """
        if plot_type == "waterfall":
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            
        elif plot_type == "force":
            # Force plot returns HTML, convert to image
            fig = plt.figure(figsize=(12, 3))
            shap.plots.force(shap_values[0], show=False, matplotlib=True)
            plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Local explanation plot saved to {save_path}")
        
        return fig
    
    def explain_prediction(
        self,
        X_single: pd.DataFrame,
        return_plot: bool = True
    ) -> Dict:
        """
        Complete explanation for single prediction
        
        Args:
            X_single: Single patient features (should include ALL features model was trained on)
            return_plot: Whether to generate plot
            
        Returns:
            Dictionary with explanation details
        """
        # Ensure X_single has all features the model expects
        # The model was trained with all features (including leakage), so SHAP needs all features too
        # But for prediction, we remove leakage features
        
        # Get prediction (remove leakage for actual prediction)
        leakage = ['weak_score', 'high_risk_drug', 'faers_adr_rate', 'faers_severe_rate']
        X_safe = X_single.drop(columns=[c for c in leakage if c in X_single.columns], errors='ignore')
        
        dtest = xgb.DMatrix(X_safe)
        pred_proba = self.model.predict(dtest)[0]
        
        # For SHAP, use X_single with ALL features (model expects all features for SHAP)
        # But ensure feature alignment first
        X_for_shap = X_single.copy()
        
        # Get SHAP explanation (with all features)
        top_contributors, shap_values = self.get_local_explanation(X_for_shap, top_n=10)
        
        # Format contributors
        contributors_formatted = []
        for feature, shap_val in top_contributors:
            direction = "↑" if shap_val > 0 else "↓"
            contributors_formatted.append({
                'feature': feature,
                'shap_value': float(shap_val),
                'direction': direction,
                'magnitude': abs(float(shap_val))
            })
        
        # Create explanation text
        top_3 = [c['feature'] for c in contributors_formatted[:3]]
        explanation_text = f"Risk driven by: {', '.join(top_3)}"
        
        result = {
            'prediction': float(pred_proba),
            'top_contributors': contributors_formatted,
            'explanation_text': explanation_text,
            'base_value': float(shap_values.base_values[0])
        }
        
        # Add plot if requested
        if return_plot:
            fig = self.plot_local_explanation(shap_values, plot_type="waterfall")
            result['plot'] = fig
        
        return result
    
    def batch_explain(
        self,
        X: pd.DataFrame,
        save_summary: bool = True,
        summary_path: str = "reports/shap_summary.png"
    ) -> Dict:
        """
        Explain multiple predictions
        
        Args:
            X: Multiple patient features
            save_summary: Whether to save summary plot
            summary_path: Path for summary plot
            
        Returns:
            Dictionary with batch explanation results
        """
        print(f"\nExplaining {len(X)} predictions...")
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X)
        
        # Get global importance
        importance_df = self.get_global_importance(X, top_n=20, save_path=None)
        
        # Save summary plot if requested
        if save_summary:
            self.plot_global_importance(shap_values, summary_path)
        
        result = {
            'num_samples': len(X),
            'global_importance': importance_df.to_dict('records'),
            'shap_values': shap_values
        }
        
        return result


def main():
    """
    Main function to create and save SHAP explainer
    """
    print("\n" + "="*60)
    print("CREATING SHAP EXPLAINER")
    print("="*60 + "\n")
    
    # Load model
    try:
        model = load_model("models/xgb_adr_model.json")
        print("✓ Model loaded")
    except FileNotFoundError:
        print("Error: Model not found. Please train the model first.")
        return
    
    # Load test data for background
    try:
        x_path = get_data_path("X_features.csv")
        X_test = pd.read_csv(x_path)
        print(f"✓ Loaded {len(X_test)} samples for background data")
        
        # Use subset for background
        X_background = X_test.sample(min(100, len(X_test)), random_state=42)
        
    except FileNotFoundError:
        print("Warning: Could not load feature data. Creating explainer without background.")
        X_background = None
    
    # Create explainer
    explainer = SHAPExplainer(model=model)
    explainer.create_explainer(X_background)
    
    # Save explainer
    explainer.save_explainer()
    
    # Compute and save global importance
    if X_background is not None:
        explainer.get_global_importance(
            X_background, 
            top_n=20,
            save_path="reports/shap_global_importance.png"
        )
    
    print("\n" + "="*60)
    print("SHAP EXPLAINER READY")
    print("="*60 + "\n")
    
    return explainer


if __name__ == "__main__":
    main()

