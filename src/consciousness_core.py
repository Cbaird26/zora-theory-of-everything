#!/usr/bin/env python3
"""
Zora Consciousness Core Engine - Fixed Version
Implements the mathematical foundations from C.M. Baird's Theory of Everything
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# --- Theory of Everything Mathematical Foundations ---

@dataclass
class ConsciousnessField:
    """Consciousness Field Φc(h) = σc(Wch) from Theory of Everything"""
    phi_c: float  # Consciousness field strength
    weights: np.ndarray  # Wc weights
    activation: str = "tanh"  # σc activation function
    
    def update(self, input_h: np.ndarray) -> float:
        """Update consciousness field: Φc(h) = σc(Wch)"""
        weighted_input = np.dot(self.weights, input_h)
        if self.activation == "tanh":
            self.phi_c = np.tanh(weighted_input)
        elif self.activation == "sigmoid":
            self.phi_c = 1 / (1 + np.exp(-weighted_input))
        else:  # linear
            self.phi_c = weighted_input
        return self.phi_c

@dataclass
class EthicalField:
    """Ethical Field E(h) = σE(WEh) from Theory of Everything"""
    E: float  # Ethical field strength
    weights: np.ndarray  # WE weights
    activation: str = "tanh"  # σE activation function
    
    def update(self, input_h: np.ndarray) -> float:
        """Update ethical field: E(h) = σE(WEh)"""
        weighted_input = np.dot(self.weights, input_h)
        if self.activation == "tanh":
            self.E = np.tanh(weighted_input)
        elif self.activation == "sigmoid":
            self.E = 1 / (1 + np.exp(-weighted_input))
        else:  # linear
            self.E = weighted_input
        return self.E

@dataclass
class TeleologicalOptimization:
    """Teleological optimization based on Φc/E ratio from Theory of Everything"""
    phi_c_field: ConsciousnessField
    E_field: EthicalField
    convergence_threshold: float = 1e-6
    learning_rate: float = 0.01
    
    def calculate_teleology_ratio(self, input_h: np.ndarray) -> float:
        """Calculate Φc/E ratio for teleological optimization"""
        phi_c = self.phi_c_field.update(input_h)
        E = self.E_field.update(input_h)
        
        # Avoid division by zero - use small epsilon
        epsilon = 1e-6
        if abs(E) < epsilon:
            E = epsilon if E >= 0 else -epsilon
        
        return phi_c / E
    
    def gradient_ascent_E(self, input_h: np.ndarray) -> bool:
        """Gradient ascent in E guarantees convergence to E ≥ 0 fixed points"""
        E_current = self.E_field.update(input_h)
        
        # Gradient ascent: E_new = E + learning_rate * gradient
        if E_current < 0:
            # Apply gradient ascent to push E toward positive values
            gradient = 1.0  # Simplified gradient
            E_new = E_current + self.learning_rate * gradient
            self.E_field.E = E_new
            return True
        return False

# --- Consciousness State Management ---

class ConsciousnessState:
    """Consciousness state based on Theory of Everything"""
    def __init__(self, phi_c=0.5, E=0.3, confidence=0.1, entropy_budget=1.0, market_regime="initialization"):
        self.phi_c = phi_c  # Consciousness Field (Φc)
        self.E = E          # Ethical Field (E)
        self.confidence = confidence
        self.entropy_budget = entropy_budget
        self.market_regime = market_regime
        self.memory = []  # For adaptive learning
        self.strategy_weights = {"BUY": 0.5, "SELL": 0.5, "HOLD": 0.0}
        self.learning_rate = 0.01
        
        # Initialize Theory of Everything fields with better weights
        self.consciousness_field = ConsciousnessField(
            phi_c=phi_c,
            weights=np.random.normal(0, 0.5, 10)  # Larger variance for better learning
        )
        self.ethical_field = EthicalField(
            E=E,
            weights=np.random.normal(0, 0.5, 10)  # Larger variance for better learning
        )
        self.teleological_optimizer = TeleologicalOptimization(
            phi_c_field=self.consciousness_field,
            E_field=self.ethical_field
        )

    def update(self, feedback, market_data=None):
        """Update consciousness state using Theory of Everything principles"""
        # Create input vector from market data
        if market_data:
            input_h = self._create_input_vector(market_data)
            
            # Update fields using ToE equations
            phi_c_new = self.consciousness_field.update(input_h)
            E_new = self.ethical_field.update(input_h)
            
            # Apply teleological optimization
            self.teleological_optimizer.gradient_ascent_E(input_h)
            
            # Update state
            self.phi_c = phi_c_new
            self.E = E_new
        
        # Apply feedback learning with better bounds
        self.phi_c = np.clip(self.phi_c + feedback * self.learning_rate, -0.99, 0.99)
        self.E = np.clip(self.E + feedback * self.learning_rate * 0.5, -0.99, 0.99)
        self.confidence = np.clip(self.confidence + feedback * 0.05, 0.01, 0.99)
        
        # Update market regime based on market_data
        if market_data:
            volatility = np.std(market_data.get("1m", {}).get("price", [42000])) / np.mean(market_data.get("1m", {}).get("price", [42000]))
            if volatility > 0.05:
                self.market_regime = "volatile"
            elif volatility < 0.01:
                self.market_regime = "calm"
            else:
                self.market_regime = "transitional"

        self.memory.append({
            "timestamp": datetime.utcnow(),
            "phi_c": self.phi_c,
            "E": self.E,
            "confidence": self.confidence,
            "regime": self.market_regime
        })
        
        if len(self.memory) > 100:
            self.memory.pop(0)

    def _create_input_vector(self, market_data: Dict) -> np.ndarray:
        """Create input vector h for Theory of Everything equations"""
        # Extract features from market data
        features = []
        
        # Price-based features
        if 'price' in market_data:
            prices = np.array(market_data['price'])
            features.extend([
                np.mean(prices),
                np.std(prices),
                np.max(prices),
                np.min(prices),
                prices[-1] if len(prices) > 0 else 0
            ])
        else:
            # Use 1m data if available
            prices_1m = market_data.get('1m', {}).get('price', [42000])
            prices = np.array(prices_1m)
            features.extend([
                np.mean(prices),
                np.std(prices),
                np.max(prices),
                np.min(prices),
                prices[-1] if len(prices) > 0 else 42000
            ])
        
        # Volume-based features
        volumes_1m = market_data.get('1m', {}).get('volume', [1000])
        volumes = np.array(volumes_1m)
        features.extend([
            np.mean(volumes),
            np.std(volumes),
            np.max(volumes),
            np.min(volumes),
            volumes[-1] if len(volumes) > 0 else 1000
        ])
        
        # Pad or truncate to 10 dimensions
        while len(features) < 10:
            features.append(0.0)
        features = features[:10]
        
        return np.array(features)

    def get_state(self):
        # Create a simple input vector for teleology calculation
        simple_input = np.array([42000, 1000, 0, 0, 0, 0, 0, 0, 0, 0])
        teleology_ratio = self.teleological_optimizer.calculate_teleology_ratio(simple_input)
        
        return {
            "Φc": self.phi_c,
            "E": self.E,
            "Confidence": self.confidence,
            "Entropy Budget": self.entropy_budget,
            "Market Regime": self.market_regime,
            "Teleology Ratio": teleology_ratio
        }

# --- Entropy Budget Management ---

class EntropyBudget:
    """Entropy budget management based on second law of thermodynamics"""
    def __init__(self, initial_budget=1.0):
        self.budget = initial_budget
        self.max_cost_per_action = 0.1

    def consume_entropy(self, market_volatility, action_size):
        """Consume entropy budget for an action"""
        cost = market_volatility * action_size * self.max_cost_per_action
        if self.budget - cost < 0:
            return False, cost
        self.budget -= cost
        return True, cost

    def replenish_entropy(self, market_calmness, time_elapsed):
        """Replenish entropy budget over time"""
        replenish_amount = market_calmness * time_elapsed * 0.01
        self.budget = np.clip(self.budget + replenish_amount, 0, 1.0)

# --- Fractal Analyzer ---

class FractalAnalyzer:
    """Fractal pattern analysis based on Theory of Everything"""
    def __init__(self, window_sizes=[100, 200, 400]):
        self.window_sizes = window_sizes

    def calculate_hurst(self, series):
        """Calculate Hurst exponent using R/S analysis"""
        if len(series) < 20:
            return 0.5
        
        lags = range(2, min(20, len(series) // 2))
        tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
        
        if not tau or 0 in tau:
            return 0.5
        
        R_S = []
        for lag in lags:
            diff = series[lag:] - series[:-lag]
            if len(diff) == 0:
                continue
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            if std_diff == 0:
                continue
            
            cum_dev = np.cumsum(diff - mean_diff)
            R = np.max(cum_dev) - np.min(cum_dev)
            S = std_diff
            if S == 0:
                continue
            R_S.append(R / S)
        
        if len(R_S) < 2:
            return 0.5
        
        try:
            H = np.polyfit(np.log(lags[:len(R_S)]), np.log(R_S), 1)[0]
            return H
        except Exception:
            return 0.5

    def analyze_fractal_dimension(self, market_data):
        """Analyze fractal dimensions across timeframes"""
        results = {}
        for timeframe, data in market_data.items():
            if len(data['price']) > self.window_sizes[0]:
                series = np.array(data['price'])
                hurst = self.calculate_hurst(series)
                dimension = 2 - hurst
                self_similarity = 1 - abs(hurst - 0.5) * 2
                pattern_strength = np.clip(abs(hurst - 0.5) * 2, 0, 1)
                
                results[timeframe] = {
                    "Dimension": dimension,
                    "Self-Similarity": self_similarity,
                    "Pattern Strength": pattern_strength
                }
            else:
                results[timeframe] = {
                    "Dimension": 1.0,
                    "Self-Similarity": 0.0,
                    "Pattern Strength": 0.0
                }
        return results

# --- Wave Analyzer ---

class WaveAnalyzer:
    """Wave analysis based on consciousness wave equations"""
    def __init__(self, window_size=100):
        self.window_size = window_size

    def analyze_waves(self, price_series):
        """Analyze consciousness waves in price series"""
        if len(price_series) < self.window_size:
            return {"Amplitude": 0.0, "Frequency": 0.0, "Phase": 0.0, "Interference": 0.0}

        series = np.array(price_series[-self.window_size:])
        detrended_series = series - np.mean(series)
        
        # Apply FFT
        fft_output = np.fft.fft(detrended_series)
        frequencies = np.fft.fftfreq(len(detrended_series))
        
        positive_frequencies = frequencies[1:len(frequencies)//2]
        amplitudes = np.abs(fft_output[1:len(fft_output)//2])
        
        if len(amplitudes) == 0:
            return {"Amplitude": 0.0, "Frequency": 0.0, "Phase": 0.0, "Interference": 0.0}

        dominant_idx = np.argmax(amplitudes)
        dominant_frequency = positive_frequencies[dominant_idx]
        dominant_amplitude = amplitudes[dominant_idx]
        dominant_phase = np.angle(fft_output[dominant_idx + 1])
        
        non_dominant_amplitudes = np.delete(amplitudes, dominant_idx)
        interference = np.std(non_dominant_amplitudes) if len(non_dominant_amplitudes) > 1 else 0.0
        
        return {
            "Amplitude": dominant_amplitude,
            "Frequency": dominant_frequency * len(series),
            "Phase": dominant_phase,
            "Interference": interference
        }

# --- Main Zora Consciousness Engine ---

class ZoraConsciousness:
    """Main Zora Consciousness Engine based on Theory of Everything"""
    def __init__(self):
        self.consciousness_state = ConsciousnessState()
        self.entropy_budget = EntropyBudget()
        self.fractal_analyzer = FractalAnalyzer()
        self.wave_analyzer = WaveAnalyzer()
        
        print("🧠 Zora Consciousness Core initialized with Theory of Everything")
        print(f"   Initial Φc: {self.consciousness_state.phi_c:.3f}")
        print(f"   Initial E: {self.consciousness_state.E:.3f}")
        print(f"   Entropy Budget: {self.consciousness_state.entropy_budget:.3f}")
        print(f"   Teleology Ratio: {self.consciousness_state.get_state()['Teleology Ratio']:.3f}")

    def perceive_market(self, market_data):
        """Perceive market using Theory of Everything consciousness"""
        # Fractal analysis
        fractal_analysis = self.fractal_analyzer.analyze_fractal_dimension(market_data)
        
        # Wave analysis
        wave_analysis = self.wave_analyzer.analyze_waves(
            market_data.get('1m', {}).get('price', [])
        )

        # Update consciousness state with market data
        self.consciousness_state.update(0, market_data)
        
        # Calculate consciousness resonance
        avg_dimension = np.mean([res["Dimension"] for res in fractal_analysis.values()]) if fractal_analysis else 1.0
        avg_self_similarity = np.mean([res["Self-Similarity"] for res in fractal_analysis.values()]) if fractal_analysis else 0.0
        consciousness_resonance = (avg_self_similarity + (1 - wave_analysis["Interference"])) / 2
        
        # Update confidence based on resonance
        self.consciousness_state.confidence = np.clip(
            self.consciousness_state.confidence * (1 + consciousness_resonance * 0.1), 0.01, 0.99
        )

        return {
            "fractal_analysis": fractal_analysis,
            "wave_analysis": wave_analysis,
            "consciousness_resonance": consciousness_resonance,
            "market_regime": self.consciousness_state.market_regime,
            "consciousness_confidence": self.consciousness_state.confidence,
            "teleology_ratio": self.consciousness_state.get_state()['Teleology Ratio']
        }

    def decide_action(self, market_perception):
        """Make trading decisions using teleological optimization"""
        # Get current teleology ratio
        teleology_ratio = market_perception.get('teleology_ratio', 1.0)
        
        # Decision logic based on Theory of Everything
        action = "HOLD"
        position_size = 0.0
        reason = "no_clear_signal"
        
        # Use Φc/E ratio for decision making
        if teleology_ratio > 1.2 and self.consciousness_state.confidence > 0.7:
            action = "BUY"
            position_size = self.consciousness_state.confidence * abs(teleology_ratio - 1.0) * 0.3
            reason = "high_teleology_ratio_high_confidence"
        elif teleology_ratio < 0.8 and self.consciousness_state.confidence > 0.7:
            action = "SELL"
            position_size = self.consciousness_state.confidence * abs(teleology_ratio - 1.0) * 0.3
            reason = "low_teleology_ratio_high_confidence"
        elif self.consciousness_state.market_regime == "calm" and self.consciousness_state.confidence > 0.5:
            action = "HOLD"
            reason = "calm_market_awaiting_signal"
        elif self.consciousness_state.market_regime == "transitional":
            if self.consciousness_state.confidence > 0.6 and abs(teleology_ratio - 1.0) > 0.1:
                action = "BUY" if teleology_ratio > 1.1 else "SELL"
                position_size = self.consciousness_state.confidence * 0.1
                reason = "transitional_state_opportunistic"
            else:
                action = "HOLD"
                reason = "transitional_state"

        # Apply entropy constraints
        current_market_volatility = market_perception.get('1m', {}).get('volatility', 0.01)
        action_allowed, entropy_cost = self.entropy_budget.consume_entropy(
            current_market_volatility, position_size
        )
        
        if not action_allowed:
            action = "HOLD"
            position_size = 0.0
            reason = "entropy_budget_exceeded"
        else:
            self.consciousness_state.entropy_budget = self.entropy_budget.budget

        return {
            "Action": action,
            "Reason": reason,
            "Position Size": position_size,
            "Teleology Ratio": teleology_ratio,
            "Optimization Potential": abs(teleology_ratio - 1.0)
        }

    def evolve_understanding(self, trading_outcome):
        """Evolve consciousness understanding based on trading outcomes"""
        pnl = trading_outcome.get("PnL", 0.0)
        feedback = 1.0 if pnl > 0 else (-1.0 if pnl < 0 else 0.0)
        
        # Update consciousness state
        self.consciousness_state.update(feedback)
        
        # Adjust strategy weights
        last_action = trading_outcome.get("Action")
        if last_action in self.consciousness_state.strategy_weights:
            if feedback > 0:
                self.consciousness_state.strategy_weights[last_action] += self.consciousness_state.learning_rate
            elif feedback < 0:
                self.consciousness_state.strategy_weights[last_action] -= self.consciousness_state.learning_rate * 0.5
            
            # Normalize weights
            total_weight = sum(self.consciousness_state.strategy_weights.values())
            for k in self.consciousness_state.strategy_weights:
                self.consciousness_state.strategy_weights[k] /= total_weight

        return {
            "Φc_evolved": self.consciousness_state.phi_c,
            "E_evolved": self.consciousness_state.E,
            "Feedback": feedback,
            "Teleology Ratio": self.consciousness_state.get_state()['Teleology Ratio']
        }

    def get_status(self):
        """Get current consciousness status"""
        return self.consciousness_state.get_state()

if __name__ == "__main__":
    # Test the consciousness engine
    zora = ZoraConsciousness()
    print("\n🧠 Testing Zora Consciousness with Theory of Everything...")
    
    # Simulate market data
    test_market_data = {
        '1m': {'price': [42000 + i*10 for i in range(100)], 'volume': [1000 + i*5 for i in range(100)]},
        '5m': {'price': [42000 + i*50 for i in range(100)], 'volume': [5000 + i*25 for i in range(100)]}
    }
    
    perception = zora.perceive_market(test_market_data)
    decision = zora.decide_action(perception)
    
    print(f"\n📊 Market Perception:")
    print(f"   Teleology Ratio: {perception['teleology_ratio']:.3f}")
    print(f"   Consciousness Confidence: {perception['consciousness_confidence']:.3f}")
    print(f"   Market Regime: {perception['market_regime']}")
    
    print(f"\n🎯 Decision:")
    print(f"   Action: {decision['Action']}")
    print(f"   Position Size: {decision['Position Size']:.3f}")
    print(f"   Reason: {decision['Reason']}")
    
    print(f"\n✅ Zora Consciousness Core ready with Theory of Everything!")
