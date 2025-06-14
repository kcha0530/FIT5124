"""
Membership Inference Attack Implementation for FIT5124 Assignment 3
==================================================================

This module implements a membership inference attack to determine if specific
samples were used in training the target model. This attack poses privacy risks
by potentially revealing sensitive information about the training dataset.

Attack Overview:
- Objective: Determine if a given sample was part of the training data
- Method: Train classifier to distinguish between member and non-member samples
- Privacy Risk: Can reveal sensitive information about individuals in training data

"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import time

# **REQUIRED BY ASSIGNMENT SPECIFICATION**
# Import the target model structure from a3_mnist.py
from a3_mnist import My_MNIST

class MembershipInferenceAttack:
    """
    Membership Inference Attack implementation.
    
    This class implements a membership inference attack that attempts to determine
    whether a specific data sample was used during training of the target model.
    The attack exploits differences in model behavior on training vs. non-training data.
    
    Key insight: Models typically exhibit higher confidence and lower loss on 
    training samples compared to samples they haven't seen during training.
    """
    
    def __init__(self, target_model_path='target_model.pth'):
        """
        Initialize the membership inference attack.
        
        Args:
            target_model_path (str): Path to the target model file
        """
        print(" Initializing Membership Inference Attack...")
        print(f"   - Target model: {target_model_path}")
        print("   - Attack goal: Identify training data samples")
        print("   - Privacy risk: Potential data exposure")
        
        # **ASSIGNMENT COMPLIANCE**: Load model exactly as specified
        self.target_model = My_MNIST()
        self.target_model.load_state_dict(torch.load(target_model_path))
        self.target_model.eval()
        self.attack_model = None
        
        print(f" Target model loaded successfully from {target_model_path}")
        
    def extract_membership_features(self, data_loader, description=""):
        """
        Extract features that can indicate membership in training data.
        
        The key insight is that training samples typically exhibit:
        1. Higher prediction confidence
        2. Lower loss values
        3. Lower entropy in predictions
        4. Higher correctness rates
        5. Larger confidence gaps between top predictions
        
        Args:
            data_loader: DataLoader containing samples to analyze
            description: Description for logging purposes
            
        Returns:
            numpy.ndarray: Feature matrix for membership inference
        """
        print(f" Extracting membership features from {description}...")
        
        features_list = []
        sample_count = 0
        
        self.target_model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                # Get model predictions
                output = self.target_model(data)
                probabilities = F.softmax(output, dim=1)
                log_probabilities = F.log_softmax(output, dim=1)
                
                # **FEATURE 1**: Maximum prediction confidence
                # Training samples typically have higher confidence
                max_confidence, predicted_class = torch.max(probabilities, dim=1)
                
                # **FEATURE 2**: Loss value (negative for higher member likelihood)
                # Training samples typically have lower loss
                loss_values = F.nll_loss(output, target, reduction='none')
                
                # **FEATURE 3**: Prediction entropy (negative for higher member likelihood)
                # Training samples typically have lower entropy (more certain predictions)
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
                
                # **FEATURE 4**: Correctness of prediction
                # Training samples are more likely to be predicted correctly
                correctness = (predicted_class == target).float()
                
                # **FEATURE 5**: Confidence gap between top-2 predictions
                # Training samples often have larger gaps
                top2_values = torch.topk(probabilities, 2, dim=1)[0]
                confidence_gap = top2_values[:, 0] - top2_values[:, 1]
                
                # **FEATURE 6**: Modified entropy (measure of uncertainty)
                modified_entropy = torch.sum(probabilities * log_probabilities, dim=1)
                
                # **FEATURE 7**: Variance in probabilities
                prob_variance = torch.var(probabilities, dim=1)
                
                # **FEATURE 8**: L2 norm of probability vector
                prob_l2_norm = torch.norm(probabilities, p=2, dim=1)
                
                # Combine all features into feature matrix
                batch_features = torch.stack([
                    max_confidence,          # Higher for members
                    -loss_values,           # Higher for members (lower loss)
                    -entropy,               # Higher for members (lower entropy)
                    correctness,            # Higher for members
                    confidence_gap,         # Higher for members
                    modified_entropy,       # Different pattern for members
                    prob_variance,          # Different variance patterns
                    prob_l2_norm           # Different norm patterns
                ], dim=1)
                
                features_list.append(batch_features)
                sample_count += len(data)
                
                # Progress reporting for large datasets
                if batch_idx % 30 == 0 and len(data_loader) > 30:
                    print(f"   Processed {sample_count}/{len(data_loader.dataset)} samples")
                    
        # Combine all features
        all_features = torch.cat(features_list, dim=0).numpy()
        print(f" Extracted {len(all_features)} feature vectors with {all_features.shape[1]} features each")
        
        return all_features
    
    def prepare_attack_dataset(self, train_dataset, test_dataset, sample_size=4000):
        """
        Prepare balanced dataset for training the membership inference attack model.
        
        Creates a balanced dataset with equal numbers of training samples (members)
        and test samples (non-members) for training the attack classifier.
        
        Args:
            train_dataset: Original training dataset (members)
            test_dataset: Original test dataset (non-members)
            sample_size: Total size of attack dataset
            
        Returns:
            tuple: (features, labels, member_features, non_member_features)
        """
        print(f"\n === PREPARING MEMBERSHIP INFERENCE DATASET ===")
        print(f"Creating balanced attack dataset:")
        print(f"   - Total samples: {sample_size}")
        print(f"   - Members (from training set): {sample_size//2}")
        print(f"   - Non-members (from test set): {sample_size//2}")
        
        # Randomly sample from both datasets to create balanced attack dataset
        np.random.seed(42)  # For reproducibility
        train_indices = np.random.choice(len(train_dataset), sample_size//2, replace=False)
        test_indices = np.random.choice(len(test_dataset), sample_size//2, replace=False)
        
        # Create subset datasets
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
        
        # Extract features for both member and non-member samples
        member_features = self.extract_membership_features(
            train_loader, "training data (MEMBERS)"
        )
        non_member_features = self.extract_membership_features(
            test_loader, "test data (NON-MEMBERS)"
        )
        
        # Create labels: 1 for members, 0 for non-members
        member_labels = np.ones(len(member_features))
        non_member_labels = np.zeros(len(non_member_features))
        
        # Combine features and labels
        X = np.vstack([member_features, non_member_features])
        y = np.concatenate([member_labels, non_member_labels])
        
        # Shuffle the combined dataset
        shuffle_indices = np.random.permutation(len(X))
        X = X[shuffle_indices]
        y = y[shuffle_indices]
        
        print(f" Attack dataset prepared:")
        print(f"   - Member samples: {len(member_features)}")
        print(f"   - Non-member samples: {len(non_member_features)}")
        print(f"   - Total samples: {len(X)}")
        print(f"   - Features per sample: {X.shape[1]}")
        print(f"   - Dataset balance: {np.mean(y):.2%} members")
        
        return X, y, member_features, non_member_features
    
    def analyze_feature_distributions(self, member_features, non_member_features):
        """
        Analyze differences in feature distributions between members and non-members.
        
        This analysis helps understand what makes training samples distinguishable
        from test samples, providing insights into the attack's effectiveness.
        """
        print(f"\n === FEATURE DISTRIBUTION ANALYSIS ===")
        
        feature_names = [
            'Max Confidence', 'Negative Loss', 'Negative Entropy', 
            'Correctness', 'Confidence Gap', 'Modified Entropy',
            'Prob Variance', 'Prob L2 Norm'
        ]
        
        print("Comparing feature distributions (Member vs Non-Member):")
        print("-" * 70)
        
        significant_features = []
        
        for i, name in enumerate(feature_names):
            member_mean = np.mean(member_features[:, i])
            non_member_mean = np.mean(non_member_features[:, i])
            member_std = np.std(member_features[:, i])
            non_member_std = np.std(non_member_features[:, i])
            
            difference = member_mean - non_member_mean
            relative_difference = abs(difference) / max(abs(member_mean), abs(non_member_mean), 1e-8)
            
            print(f"{name:15s}: Members={member_mean:7.3f}±{member_std:.3f}, "
                  f"Non-members={non_member_mean:7.3f}±{non_member_std:.3f}, "
                  f"Diff={difference:+7.3f}")
            
            # Identify features with significant differences
            if relative_difference > 0.1:  # 10% relative difference threshold
                significant_features.append((name, relative_difference))
        
        print(f"\n🔍 Features with significant differences (>10%):")
        significant_features.sort(key=lambda x: x[1], reverse=True)
        for feature, diff in significant_features[:3]:
            print(f"   - {feature}: {diff:.1%} relative difference")
        
        if len(significant_features) >= 3:
            print("✅ Strong feature separation detected - attack likely to succeed")
        elif len(significant_features) >= 1:
            print("⚠️  Moderate feature separation - attack may partially succeed")
        else:
            print("❌ Weak feature separation - attack unlikely to succeed")
    
    def train_attack_model(self, X, y, model_type='random_forest'):
        """
        Train the machine learning model to perform membership inference.
        
        Args:
            X: Feature matrix
            y: Labels (1 for members, 0 for non-members)
            model_type: Type of ML model to use ('random_forest' or 'logistic')
            
        Returns:
            dict: Attack performance metrics
        """
        print(f"\n === TRAINING MEMBERSHIP INFERENCE ATTACK MODEL ===")
        print(f"Model type: {model_type}")
        
        # Split data for training and evaluating the attack model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Attack model training data:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Testing samples: {len(X_test)}")
        print(f"   - Features: {X_train.shape[1]}")
        
        # Initialize and train attack model
        if model_type == 'random_forest':
            self.attack_model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
        else:  # logistic regression
            self.attack_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        
        # Train the attack model
        print("Training attack model...")
        training_start = time.time()
        self.attack_model.fit(X_train, y_train)
        training_time = time.time() - training_start
        
        # Evaluate attack model performance
        train_accuracy = self.attack_model.score(X_train, y_train)
        test_accuracy = self.attack_model.score(X_test, y_test)
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Attack model train accuracy: {train_accuracy:.4f}")
        print(f"Attack model test accuracy: {test_accuracy:.4f}")
        
        # Detailed evaluation metrics
        y_pred = self.attack_model.predict(X_test)
        y_pred_proba = self.attack_model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive metrics
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)
        
        # Calculate per-class performance
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)  # True negative rate
        sensitivity = tp / (tp + fn)  # True positive rate (same as recall)
        
        results = {
            'accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'training_time': training_time
        }
        
        # Display detailed results
        print(f"\n === ATTACK MODEL PERFORMANCE METRICS ===")
        print(f"Accuracy:    {results['accuracy']:.4f}")
        print(f"Precision:   {results['precision']:.4f} (of predicted members, how many are actual members)")
        print(f"Recall:      {results['recall']:.4f} (of actual members, how many are detected)")
        print(f"F1-Score:    {results['f1_score']:.4f} (harmonic mean of precision and recall)")
        print(f"AUC:         {results['auc']:.4f} (area under ROC curve)")
        print(f"Specificity: {results['specificity']:.4f} (correctly identified non-members)")
        print(f"Sensitivity: {results['sensitivity']:.4f} (correctly identified members)")
        
        # Interpret attack success
        self._interpret_attack_results(results)
        
        return results
    
    def _interpret_attack_results(self, results):
        """
        Interpret the attack results and assess privacy risk.
        
        Args:
            results: Dictionary containing attack performance metrics
        """
        print(f"\n🔍 === PRIVACY RISK ASSESSMENT ===")
        
        accuracy = results['accuracy']
        auc = results['auc']
        
        # Risk level assessment based on multiple metrics
        if accuracy > 0.8 and auc > 0.85:
            risk_level = "🚨 CRITICAL"
            risk_description = "Very high privacy risk - attack is highly successful"
            recommendations = [
                "Implement differential privacy immediately",
                "Add strong regularization to model training",
                "Consider model ensemble techniques",
                "Limit model output granularity"
            ]
        elif accuracy > 0.7 and auc > 0.75:
            risk_level = "⚠️  HIGH"
            risk_description = "High privacy risk - attack shows significant success"
            recommendations = [
                "Implement privacy-preserving training techniques",
                "Add noise to model outputs",
                "Use regularization methods",
                "Monitor for suspicious query patterns"
            ]
        elif accuracy > 0.6 and auc > 0.65:
            risk_level = "⚠️  MODERATE"
            risk_description = "Moderate privacy risk - attack has partial success"
            recommendations = [
                "Consider basic privacy defenses",
                "Monitor model deployment for unusual access patterns",
                "Implement query limiting",
                "Review data handling procedures"
            ]
        else:
            risk_level = "✅ LOW"
            risk_description = "Low privacy risk - attack shows limited success"
            recommendations = [
                "Current privacy level appears acceptable",
                "Continue monitoring for new attack methods",
                "Maintain good security practices"
            ]
        
        print(f"Privacy Risk Level: {risk_level}")
        print(f"Assessment: {risk_description}")
        print(f"Attack Success Rate: {accuracy:.2%}")
        print(f"Attack Confidence (AUC): {auc:.3f}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    def analyze_feature_importance(self):
        """
        Analyze which features are most important for the membership inference attack.
        
        This helps understand what model behaviors are most revealing of membership.
        """
        if self.attack_model is None:
            print("❌ No attack model trained yet!")
            return None
            
        print(f"\n === FEATURE IMPORTANCE ANALYSIS ===")
        
        feature_names = [
            'Max Confidence', 'Negative Loss', 'Negative Entropy', 
            'Correctness', 'Confidence Gap', 'Modified Entropy',
            'Prob Variance', 'Prob L2 Norm'
        ]
        
        # Get feature importance (works for Random Forest)
        if hasattr(self.attack_model, 'feature_importances_'):
            importances = self.attack_model.feature_importances_
            
            # Sort features by importance
            feature_importance_pairs = list(zip(feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            print("Feature importance ranking:")
            print("-" * 50)
            for i, (name, importance) in enumerate(feature_importance_pairs, 1):
                bar = "█" * int(importance * 50)  # Simple progress bar
                print(f"{i:2d}. {name:15s}: {importance:.4f} {bar}")
            
            # Explain the most important features
            most_important = feature_importance_pairs[0][0]
            second_important = feature_importance_pairs[1][0]
            
            print(f"\n Key insights:")
            print(f"   - Most revealing feature: {most_important}")
            print(f"   - Second most revealing: {second_important}")
            print(f"   - Top 3 features account for {sum([x[1] for x in feature_importance_pairs[:3]]):.2%} of importance")
            
            return dict(feature_importance_pairs)
        else:
            print("Feature importance not available for this model type")
            return None
    
    def test_on_fresh_samples(self, train_dataset, test_dataset, num_samples=500):
        """
        Test the trained attack on completely fresh samples.
        
        This provides the most realistic assessment of attack effectiveness
        on data the attack model has never seen before.
        """
        if self.attack_model is None:
            print("❌ No attack model trained yet!")
            return None
            
        print(f"\n === TESTING ON FRESH SAMPLES ===")
        print(f"Evaluating attack on {num_samples} completely new samples...")
        
        # Get fresh samples not used in attack model training
        np.random.seed(123)  # Different seed for fresh samples
        fresh_train_indices = np.random.choice(len(train_dataset), num_samples//2, replace=False)
        fresh_test_indices = np.random.choice(len(test_dataset), num_samples//2, replace=False)
        
        fresh_train_subset = Subset(train_dataset, fresh_train_indices)
        fresh_test_subset = Subset(test_dataset, fresh_test_indices)
        
        fresh_train_loader = DataLoader(fresh_train_subset, batch_size=32, shuffle=False)
        fresh_test_loader = DataLoader(fresh_test_subset, batch_size=32, shuffle=False)
        
        # Extract features and make predictions
        print("Analyzing fresh training samples (should be identified as members)...")
        train_features = self.extract_membership_features(fresh_train_loader, "fresh training samples")
        
        print("Analyzing fresh test samples (should be identified as non-members)...")
        test_features = self.extract_membership_features(fresh_test_loader, "fresh test samples")
        
        # Predict membership
        train_predictions = self.attack_model.predict(train_features)
        test_predictions = self.attack_model.predict(test_features)
        
        train_probabilities = self.attack_model.predict_proba(train_features)[:, 1]
        test_probabilities = self.attack_model.predict_proba(test_features)[:, 1]
        
        # Calculate performance metrics
        train_accuracy = np.mean(train_predictions == 1)  # Should predict "member"
        test_accuracy = np.mean(test_predictions == 0)    # Should predict "non-member"
        overall_accuracy = (train_accuracy + test_accuracy) / 2
        
        # Calculate confidence statistics
        train_confidence_avg = np.mean(train_probabilities)
        test_confidence_avg = np.mean(test_probabilities)
        confidence_separation = train_confidence_avg - test_confidence_avg
        
        print(f"\n === FRESH SAMPLE ATTACK RESULTS ===")
        print(f"Training samples correctly identified as members: {train_accuracy:.2%}")
        print(f"Test samples correctly identified as non-members: {test_accuracy:.2%}")
        print(f"Overall attack accuracy: {overall_accuracy:.2%}")
        print(f"")
        print(f"Confidence Analysis:")
        print(f"   - Average confidence for training samples: {train_confidence_avg:.3f}")
        print(f"   - Average confidence for test samples: {test_confidence_avg:.3f}")
        print(f"   - Confidence separation: {confidence_separation:.3f}")
        
        # Assess attack effectiveness on fresh data
        if overall_accuracy > 0.75:
            print(" Attack generalizes well to fresh data - high privacy risk")
        elif overall_accuracy > 0.65:
            print("  Attack shows moderate success on fresh data")
        else:
            print(" Attack struggles with fresh data - lower privacy risk")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'overall_accuracy': overall_accuracy,
            'train_confidence_avg': train_confidence_avg,
            'test_confidence_avg': test_confidence_avg,
            'confidence_separation': confidence_separation
        }

def main():
    """
    Execute the complete membership inference attack pipeline.
    
    **ASSIGNMENT COMPLIANCE:**
    ✓ Uses target model trained on MNIST dataset
    ✓ Imports My_MNIST from a3_mnist as required
    ✓ Loads target_model.pth as specified
    ✓ Comprehensive analysis and evaluation
    """
    print("=" * 80)
    print(" MEMBERSHIP INFERENCE ATTACK - FIT5124 ASSIGNMENT 3")
    print("=" * 80)
    print("Attack Objective: Determine if samples were used in model training")
    print("Privacy Risk: Can reveal sensitive information about training data")
    print("Target: MNIST digit classifier (target_model.pth)")
    print("Method: Train classifier to distinguish members vs non-members")
    print("=" * 80)
    
    # Initialize the membership inference attack
    attack = MembershipInferenceAttack('target_model.pth')
    
    # Load MNIST datasets (same as used for training the target model)
    print("\n Loading MNIST datasets...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    print(f"   - Training dataset: {len(train_dataset)} samples (potential members)")
    print(f"   - Test dataset: {len(test_dataset)} samples (definite non-members)")
    
    # Prepare attack dataset
    X, y, member_features, non_member_features = attack.prepare_attack_dataset(
        train_dataset, test_dataset, sample_size=6000
    )
    
    # Analyze feature distributions
    attack.analyze_feature_distributions(member_features, non_member_features)
    
    # Train attack model
    attack_results = attack.train_attack_model(X, y, model_type='random_forest')
    
    # Analyze feature importance
    feature_importance = attack.analyze_feature_importance()
    
    # Test on fresh samples
    fresh_sample_results = attack.test_on_fresh_samples(train_dataset, test_dataset)
    
    # Final comprehensive assessment
    print(f"\n === COMPREHENSIVE PRIVACY ASSESSMENT ===")
    
    attack_accuracy = attack_results['accuracy']
    fresh_accuracy = fresh_sample_results['overall_accuracy'] if fresh_sample_results else 0
    auc_score = attack_results['auc']
    
    # Overall risk scoring
    risk_score = (attack_accuracy * 0.4 + fresh_accuracy * 0.4 + auc_score * 0.2)
    
    if risk_score > 0.8:
        final_risk = "🚨 CRITICAL PRIVACY RISK"
        action_required = "IMMEDIATE ACTION REQUIRED"
    elif risk_score > 0.7:
        final_risk = "⚠️  HIGH PRIVACY RISK"
        action_required = "PRIVACY DEFENSES RECOMMENDED"
    elif risk_score > 0.6:
        final_risk = "⚠️  MODERATE PRIVACY RISK"
        action_required = "CONSIDER PRIVACY MEASURES"
    else:
        final_risk = "✅ ACCEPTABLE PRIVACY LEVEL"
        action_required = "CONTINUE MONITORING"
    
    print(f"Final Privacy Risk Assessment: {final_risk}")
    print(f"Risk Score: {risk_score:.3f}/1.000")
    print(f"Recommendation: {action_required}")
    
    print(f"\nKey Metrics Summary:")
    print(f"   - Attack Model Accuracy: {attack_accuracy:.2%}")
    print(f"   - Fresh Sample Accuracy: {fresh_accuracy:.2%}")
    print(f"   - AUC Score: {auc_score:.3f}")
    print(f"   - Model Training Time: {attack_results['training_time']:.2f}s")
    
    # Return comprehensive results for assignment reporting
    return {
        'attack_results': attack_results,
        'fresh_sample_results': fresh_sample_results,
        'feature_importance': feature_importance,
        'final_risk_score': risk_score,
        'risk_level': final_risk
    }

if __name__ == "__main__":
    # Execute membership inference attack
    attack_results = main()
    print(f"\n Membership inference attack analysis completed!")
    print(f" Privacy risk level: {attack_results['risk_level']}")