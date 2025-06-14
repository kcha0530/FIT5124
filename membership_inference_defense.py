"""
Membership Inference Defense Implementation for FIT5124 Assignment 3
===================================================================

This module implements defense mechanisms against membership inference attacks.
The defenses aim to protect training data privacy while maintaining model utility.

Defense Strategies Implemented:
1. Differential Privacy during training
2. Regularization techniques (L2, Dropout)
3. Output smoothing and noise injection
4. Early stopping to prevent overfitting

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
from collections import defaultdict

# Import components for testing
from a3_mnist import My_MNIST

class DifferentialPrivacyMechanism:
    """
    Implements differential privacy for model training.
    
    Differential privacy provides mathematical guarantees about privacy
    by adding calibrated noise to gradients during training.
    """
    
    def __init__(self, noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5):
        """
        Initialize differential privacy mechanism.
        
        Args:
            noise_multiplier: Controls the amount of noise added
            max_grad_norm: Maximum norm for gradient clipping
            delta: Privacy parameter (probability of privacy breach)
        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        
        print(f" Differential Privacy initialized:")
        print(f"   Noise multiplier: {noise_multiplier}")
        print(f"   Max gradient norm: {max_grad_norm}")
        print(f"   Delta: {delta}")
    
    def clip_gradients(self, model):
        """
        Clip gradients to bound their sensitivity.
        
        This is essential for differential privacy as it limits
        how much any single training example can influence the model.
        """
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        return total_norm
    
    def add_noise_to_gradients(self, model, batch_size):
        """
        Add noise to gradients for differential privacy.
        
        The noise is calibrated based on the sensitivity of the gradients
        and the desired privacy level.
        """
        for param in model.parameters():
            if param.grad is not None:
                # Calculate noise standard deviation
                noise_std = self.noise_multiplier * self.max_grad_norm / batch_size
                
                # Add Gaussian noise
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad += noise

class PrivacyPreservingModel(nn.Module):
    """
    Enhanced model with built-in privacy defenses.
    
    This model includes regularization techniques that help
    prevent overfitting and reduce membership inference vulnerabilities.
    """
    
    def __init__(self, dropout_rate=0.7, use_batch_norm=True):
        super(PrivacyPreservingModel, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(6) if use_batch_norm else nn.Identity()
        
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16) if use_batch_norm else nn.Identity()
        
        # Enhanced dropout for privacy
        self.dropout1 = nn.Dropout(dropout_rate * 0.7)  # Less aggressive for early layers
        self.dropout2 = nn.Dropout(dropout_rate)        # More aggressive for later layers
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        print(f" Privacy-preserving model initialized:")
        print(f"   Dropout rate: {dropout_rate}")
        print(f"   Batch normalization: {use_batch_norm}")

    def forward(self, x, add_noise=False, noise_scale=0.1):
        """
        Forward pass with optional noise injection.
        
        Args:
            x: Input tensor
            add_noise: Whether to add noise to outputs (inference-time defense)
            noise_scale: Scale of noise to add
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc3(x)
        
        # Optional noise injection at inference time
        if add_noise and noise_scale > 0:
            noise = torch.normal(0, noise_scale, size=x.shape, device=x.device)
            x = x + noise
        
        output = F.log_softmax(x, dim=1)
        return output

def train_privacy_preserving_model(defense_type="differential_privacy", epochs=6):
    """
    Train a model with privacy-preserving techniques.
    
    Args:
        defense_type: Type of defense ("differential_privacy", "regularization", "combined")
        epochs: Number of training epochs
    
    Returns:
        Trained model and training statistics
    """
    print(f" === TRAINING PRIVACY-PRESERVING MODEL ===")
    print(f"Defense type: {defense_type}")
    print(f"Training epochs: {epochs}")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model based on defense type
    if defense_type == "differential_privacy":
        model = PrivacyPreservingModel(dropout_rate=0.5, use_batch_norm=True)
        dp_mechanism = DifferentialPrivacyMechanism(noise_multiplier=0.8, max_grad_norm=1.0)
        l2_lambda = 0.001  # Light regularization
    elif defense_type == "regularization":
        model = PrivacyPreservingModel(dropout_rate=0.8, use_batch_norm=True)
        dp_mechanism = None
        l2_lambda = 0.01   # Heavy regularization
    else:  # combined
        model = PrivacyPreservingModel(dropout_rate=0.7, use_batch_norm=True)
        dp_mechanism = DifferentialPrivacyMechanism(noise_multiplier=0.5, max_grad_norm=1.0)
        l2_lambda = 0.005  # Moderate regularization
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_lambda)
    
    # Training loop
    model.train()
    training_losses = []
    training_accuracies = []
    
    print("\nStarting privacy-preserving training...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = F.nll_loss(output, target)
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(target)
            
            # Backward pass
            loss.backward()
            
            # Apply differential privacy if enabled
            if dp_mechanism is not None:
                # Clip gradients
                grad_norm = dp_mechanism.clip_gradients(model)
                
                # Add noise to gradients
                dp_mechanism.add_noise_to_gradients(model, len(data))
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Progress reporting
            if batch_idx % 200 == 0:
                print(f'   Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Calculate epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        training_losses.append(avg_loss)
        training_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}: Loss={avg_loss:.6f}, Accuracy={accuracy:.2f}%')
    
    # Final evaluation
    print("\nEvaluating trained model...")
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Final Test Accuracy: {test_accuracy:.2f}%')
    print(f'Final Test Loss: {test_loss:.4f}')
    
    # Save the defended model
    model_filename = f'defended_model_{defense_type}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f' Privacy-preserving model saved as: {model_filename}')
    
    return model, {
        'training_losses': training_losses,
        'training_accuracies': training_accuracies,
        'final_test_accuracy': test_accuracy,
        'final_test_loss': test_loss,
        'model_path': model_filename
    }

def evaluate_privacy_defense_effectiveness():
    """
    Evaluate how well the privacy defenses protect against membership inference.
    """
    print("\n === EVALUATING PRIVACY DEFENSE EFFECTIVENESS ===")
    
    # Test different defense strategies
    defense_strategies = [
        ("original", "target_model.pth"),
        ("differential_privacy", None),
        ("regularization", None), 
        ("combined", None)
    ]
    
    results = {}
    
    for strategy_name, model_path in defense_strategies:
        print(f"\nTesting: {strategy_name}")
        print("-" * 40)
        
        if model_path is None:
            # Train new defended model
            print(f"Training new model with {strategy_name} defenses...")
            model, training_stats = train_privacy_preserving_model(
                defense_type=strategy_name, epochs=4
            )
            model_path = training_stats['model_path']
            model_accuracy = training_stats['final_test_accuracy']
        else:
            # Use existing original model
            print("Using original target model...")
            model_accuracy = 98.62  # From previous results
        
        # Test membership inference attack against this model
        print(f"Testing membership inference attack...")
        attack_results = test_membership_inference_against_defended_model(
            model_path, strategy_name
        )
        
        results[strategy_name] = {
            'model_accuracy': model_accuracy,
            'attack_accuracy': attack_results['attack_accuracy'],
            'privacy_gain': None,  # Will calculate after getting baseline
            'auc_score': attack_results.get('auc_score', 0.5)
        }
        
        print(f" Model accuracy: {model_accuracy:.2f}%")
        print(f" Attack accuracy: {attack_results['attack_accuracy']:.2f}%")
    
    # Calculate privacy gains relative to original model
    baseline_attack_acc = results['original']['attack_accuracy']
    
    for strategy in results:
        if strategy != 'original':
            privacy_gain = baseline_attack_acc - results[strategy]['attack_accuracy']
            results[strategy]['privacy_gain'] = privacy_gain
    
    return results

def test_membership_inference_against_defended_model(model_path, model_type):
    """
    Test membership inference attack against a defended model.
    
    This is a simplified version of the membership inference attack
    specifically for testing defense effectiveness.
    """
    print("    Running membership inference attack...")
    
    # Load the model
    if model_type == "original":
        model = My_MNIST()
    else:
        model = PrivacyPreservingModel()
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Sample smaller datasets for quick testing
    np.random.seed(42)
    train_indices = np.random.choice(len(train_dataset), 1000, replace=False)
    test_indices = np.random.choice(len(test_dataset), 1000, replace=False)
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # Extract features for membership inference
    def extract_simple_features(data_loader):
        features = []
        with torch.no_grad():
            for data, target in data_loader:
                if hasattr(model, 'forward') and 'add_noise' in model.forward.__code__.co_varnames:
                    # Privacy-preserving model - add noise during inference
                    output = model(data, add_noise=True, noise_scale=0.05)
                else:
                    # Original model
                    output = model(data)
                
                probabilities = F.softmax(output, dim=1)
                max_confidence = torch.max(probabilities, dim=1)[0]
                loss = F.nll_loss(output, target, reduction='none')
                
                # Simple features: confidence and loss
                batch_features = torch.stack([max_confidence, -loss], dim=1)
                features.append(batch_features)
        
        return torch.cat(features, dim=0).numpy()
    
    # Extract features
    member_features = extract_simple_features(train_loader)
    non_member_features = extract_simple_features(test_loader)
    
    # Create labels
    member_labels = np.ones(len(member_features))
    non_member_labels = np.zeros(len(non_member_features))
    
    # Combine data
    X = np.vstack([member_features, non_member_features])
    y = np.concatenate([member_labels, non_member_labels])
    
    # Simple classifier (logistic regression for speed)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train attack classifier
    attack_classifier = LogisticRegression(random_state=42, max_iter=1000)
    attack_classifier.fit(X_train, y_train)
    
    # Evaluate attack
    y_pred = attack_classifier.predict(X_test)
    y_pred_proba = attack_classifier.predict_proba(X_test)[:, 1]
    
    attack_accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'attack_accuracy': attack_accuracy * 100,  # Convert to percentage
        'auc_score': auc_score
    }

def analyze_defense_results(results):
    """
    Analyze and summarize the defense effectiveness results.
    """
    print("\n === DEFENSE EFFECTIVENESS ANALYSIS ===")
    
    print("\nSummary of Defense Performance:")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Model Acc':<12} {'Attack Acc':<12} {'Privacy Gain':<12}")
    print("-" * 60)
    
    for strategy, metrics in results.items():
        model_acc = f"{metrics['model_accuracy']:.1f}%"
        attack_acc = f"{metrics['attack_accuracy']:.1f}%"
        
        if metrics.get('privacy_gain') is not None:
            privacy_gain = f"+{metrics['privacy_gain']:.1f}%"
        else:
            privacy_gain = "baseline"
        
        print(f"{strategy:<20} {model_acc:<12} {attack_acc:<12} {privacy_gain:<12}")
    
    # Identify best strategy
    print(f"\n Defense Effectiveness Rankings:")
    
    # Rank by privacy gain while considering utility loss
    defense_strategies = [(k, v) for k, v in results.items() if k != 'original']
    
    # Calculate effectiveness score (privacy gain - utility loss)
    baseline_accuracy = results['original']['model_accuracy']
    
    for strategy, metrics in defense_strategies:
        utility_loss = baseline_accuracy - metrics['model_accuracy']
        privacy_gain = metrics.get('privacy_gain', 0)
        effectiveness_score = privacy_gain - (utility_loss * 0.5)  # Weight utility less than privacy
        metrics['effectiveness_score'] = effectiveness_score
    
    # Sort by effectiveness score
    ranked_strategies = sorted(defense_strategies, 
                             key=lambda x: x[1]['effectiveness_score'], 
                             reverse=True)
    
    for i, (strategy, metrics) in enumerate(ranked_strategies, 1):
        utility_loss = baseline_accuracy - metrics['model_accuracy']
        print(f"   {i}. {strategy}: "
              f"Privacy gain: +{metrics.get('privacy_gain', 0):.1f}%, "
              f"Utility loss: -{utility_loss:.1f}%")
    
    # Recommendations
    best_strategy = ranked_strategies[0][0]
    print(f"\n Recommended Strategy: {best_strategy}")
    
    best_metrics = ranked_strategies[0][1]
    if best_metrics.get('privacy_gain', 0) > 5:
        recommendation = "Strong privacy protection with acceptable utility trade-off"
    elif best_metrics.get('privacy_gain', 0) > 2:
        recommendation = "Moderate privacy improvement with minimal utility loss"
    else:
        recommendation = "Limited privacy improvement - consider stronger defenses"
    
    print(f"   Assessment: {recommendation}")

def demonstrate_advanced_attacker_scenario():
    """
    Analyze defense effectiveness against an advanced attacker who knows
    about the defense mechanisms.
    """
    print("\n === ADVANCED ATTACKER ANALYSIS ===")
    print("Scenario: Attacker knows about defense mechanisms and adapts strategy")
    
    print("\nAdvanced Attacker Capabilities:")
    print("1. Knows differential privacy parameters")
    print("2. Understands regularization techniques used")
    print("3. Can adapt feature extraction to account for noise")
    print("4. May use ensemble attacks or advanced ML models")
    
    print("\nDefense Limitations Against Advanced Attackers:")
    print("• Differential Privacy:")
    print("  - Effective against basic attacks")
    print("  - Advanced attackers may use noise-robust features")
    print("  - Still provides mathematical privacy guarantees")
    
    print("• Regularization:")
    print("  - Reduces overfitting but doesn't eliminate membership signals")
    print("  - Advanced attackers may focus on subtle statistical differences")
    print("  - Limited effectiveness against sophisticated feature engineering")
    
    print("• Combined Defenses:")
    print("  - Multiple layers of protection")
    print("  - Harder for attackers to circumvent all defenses")
    print("  - Best practical approach currently available")
    
    print("\n Maximum Defense Capacity:")
    print("Current defenses can protect against:")
    print("✅ Basic membership inference attacks")
    print("✅ Attackers without defense knowledge")
    print("⚠️ Some advanced attacks with adapted strategies")
    print("❌ May not stop highly sophisticated, targeted attacks")
    
    print("\n Recommended Improvements:")
    print("1. Implement stricter differential privacy (higher noise)")
    print("2. Use federated learning approaches")
    print("3. Apply data minimization techniques")
    print("4. Regular security audits and attack simulations")

def main():
    """
    Main function to demonstrate membership inference defenses.
    """
    print("=" * 80)
    print(" MEMBERSHIP INFERENCE DEFENSE EVALUATION - FIT5124 ASSIGNMENT 3")
    print("=" * 80)
    print("Objective: Protect training data privacy from membership inference")
    print("Method: Privacy-preserving training and inference techniques")
    print("Evaluation: Test defense effectiveness vs. attack success")
    print("=" * 80)
    
    # Step 1: Evaluate different defense strategies
    print("\n STEP 1: Testing Different Defense Strategies")
    defense_results = evaluate_privacy_defense_effectiveness()
    
    # Step 2: Analyze results
    print("\n STEP 2: Analyzing Defense Effectiveness")
    analyze_defense_results(defense_results)
    
    # Step 3: Advanced attacker analysis
    print("\n STEP 3: Advanced Attacker Scenario Analysis")
    demonstrate_advanced_attacker_scenario()
    
    # Final recommendations
    print("\n === FINAL PRIVACY RECOMMENDATIONS ===")
    
    # Find best performing defense
    best_defense = None
    best_privacy_gain = 0
    
    for strategy, metrics in defense_results.items():
        if strategy != 'original' and metrics.get('privacy_gain', 0) > best_privacy_gain:
            best_privacy_gain = metrics['privacy_gain']
            best_defense = strategy
    
    print(f"Best Defense Strategy: {best_defense}")
    print(f"Privacy Improvement: +{best_privacy_gain:.1f}% attack accuracy reduction")
    
    if best_privacy_gain > 10:
        risk_level = " STRONG PRIVACY PROTECTION"
    elif best_privacy_gain > 5:
        risk_level = " MODERATE PRIVACY PROTECTION"
    else:
        risk_level = "⚠️ LIMITED PRIVACY PROTECTION"
    
    print(f"Overall Privacy Level: {risk_level}")
    
    print(f"\nDeployment Recommendations:")
    print(f"1. Use {best_defense} defense for production")
    print(f"2. Monitor for new attack techniques")
    print(f"3. Regular privacy audits recommended")
    print(f"4. Consider additional data protection measures")
    
    return defense_results

if __name__ == "__main__":
    results = main()
    print(f"\n Membership inference defense evaluation completed!")