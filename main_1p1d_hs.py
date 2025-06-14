import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random
from collections import defaultdict
import pandas as pd

class BlackjackEnvironment:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Deal initial cards
        self.player_cards = [self.draw_card(), self.draw_card()]
        self.dealer_cards = [self.draw_card(), self.draw_card()]
        self.done = False
        return self.get_state()
    
    def draw_card(self):
        """Draw a card (1-13, where 1=Ace, 11=Jack, 12=Queen, 13=King)"""
        return random.randint(1, 13)
    
    def card_value(self, card):
        """Get the value of a card"""
        if card == 1:  # Ace
            return 11
        elif card > 10:  # Face cards
            return 10
        else:
            return card
    
    def hand_value(self, cards):
        """Calculate hand value, handling Aces"""
        value = sum(self.card_value(card) for card in cards)
        aces = sum(1 for card in cards if card == 1)
        
        # Convert Aces from 11 to 1 if needed
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
        
        return value
    
    def has_usable_ace(self, cards):
        """Check if hand has a usable ace (ace counted as 11)"""
        value = sum(self.card_value(card) for card in cards)
        aces = sum(1 for card in cards if card == 1)
        
        if aces == 0:
            return False
        
        # If we can count at least one ace as 11 without busting
        return value <= 21
    
    def get_state(self):
        """Get current state (player_sum, dealer_upcard, usable_ace)"""
        player_sum = self.hand_value(self.player_cards)
        dealer_upcard = self.card_value(self.dealer_cards[0])
        usable_ace = self.has_usable_ace(self.player_cards)
        return (player_sum, dealer_upcard, usable_ace)
    
    def step(self, action):
        """Take action: 0=Stand, 1=Hit"""
        if self.done:
            return self.get_state(), 0, True
        
        reward = 0
        
        if action == 1:  # Hit
            self.player_cards.append(self.draw_card())
            player_sum = self.hand_value(self.player_cards)
            
            if player_sum > 21:  # Player busts
                reward = -1
                self.done = True
            elif player_sum == 21:  # Player gets 21
                reward = self.dealer_play()
                self.done = True
        
        else:  # Stand
            reward = self.dealer_play()
            self.done = True
        
        return self.get_state(), reward, self.done
    
    def dealer_play(self):
        """Dealer plays according to rules (hit on soft 17)"""
        while self.hand_value(self.dealer_cards) < 17:
            self.dealer_cards.append(self.draw_card())
        
        player_sum = self.hand_value(self.player_cards)
        dealer_sum = self.hand_value(self.dealer_cards)
        
        if dealer_sum > 21:  # Dealer busts
            return 1
        elif player_sum > dealer_sum:
            return 1
        elif player_sum < dealer_sum:
            return -1
        else:
            return 0  # Tie

class BlackjackQLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(2))  # 2 actions: stand, hit
        self.env = BlackjackEnvironment()
    
    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            return np.argmax(self.q_table[state])
    
    def train(self, episodes=100000):
        """Train the Q-learning agent"""
        wins = 0
        losses = 0
        ties = 0
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.get_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                
                # Q-learning update
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.lr * td_error
                
                state = next_state
            
            # Track results
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                ties += 1
            
            # Decay epsilon
            if episode % 10000 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.95)
                print(f"Episode {episode}: Win rate = {wins/(wins+losses+ties):.3f}, Epsilon = {self.epsilon:.3f}")
        
        print(f"Training completed. Final win rate: {wins/(wins+losses+ties):.3f}")
    
    def get_policy(self):
        """Extract policy from Q-table"""
        policy = {}
        for state in self.q_table:
            policy[state] = np.argmax(self.q_table[state])
        return policy
    
    def get_action_probabilities(self):
        """Get action probabilities for visualization"""
        probs = {}
        for state in self.q_table:
            q_values = self.q_table[state]
            # Use softmax to convert Q-values to probabilities
            exp_q = np.exp(q_values - np.max(q_values))
            probs[state] = exp_q / np.sum(exp_q)
        return probs

def create_strategy_chart(agent):
    """Create strategy chart similar to the one shown"""
    policy = agent.get_policy()
    
    # Create matrices for with and without usable ace
    strategy_no_ace = np.zeros((10, 10))  # Player sums 12-21, Dealer 2-11
    strategy_ace = np.zeros((10, 10))
    
    for player_sum in range(12, 22):
        for dealer_card in range(2, 12):
            # Map dealer card 11 (Ace) to position 10
            dealer_idx = 9 if dealer_card == 11 else dealer_card - 2
            player_idx = player_sum - 12
            
            # No usable ace
            state = (player_sum, dealer_card, False)
            if state in policy:
                strategy_no_ace[player_idx, dealer_idx] = policy[state]
            
            # With usable ace
            state = (player_sum, dealer_card, True)
            if state in policy:
                strategy_ace[player_idx, dealer_idx] = policy[state]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot without usable ace
    sns.heatmap(strategy_no_ace, annot=True, fmt='.0f', cmap='RdYlGn', 
                xticklabels=['2','3','4','5','6','7','8','9','10','A'],
                yticklabels=range(12, 22), ax=ax1, cbar_kws={'label': 'Action (0=Stand, 1=Hit)'})
    ax1.set_title('Strategy: No Usable Ace')
    ax1.set_xlabel('Dealer Upcard')
    ax1.set_ylabel('Player Sum')
    
    # Plot with usable ace
    sns.heatmap(strategy_ace, annot=True, fmt='.0f', cmap='RdYlGn',
                xticklabels=['2','3','4','5','6','7','8','9','10','A'],
                yticklabels=range(12, 22), ax=ax2, cbar_kws={'label': 'Action (0=Stand, 1=Hit)'})
    ax2.set_title('Strategy: With Usable Ace')
    ax2.set_xlabel('Dealer Upcard')
    ax2.set_ylabel('Player Sum')
    
    plt.tight_layout()
    plt.show()

def create_probability_surface(agent):
    """Create 3D probability surface like the second image"""
    probs = agent.get_action_probabilities()
    
    # Create meshgrid for 3D plot
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(2, 12)
    X, Y = np.meshgrid(dealer_cards, player_sums)
    
    # Get hit probabilities (probability of action 1)
    Z_no_ace = np.zeros_like(X, dtype=float)
    Z_ace = np.zeros_like(X, dtype=float)
    
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            # No usable ace
            state = (player_sum, dealer_card, False)
            if state in probs:
                Z_no_ace[i, j] = probs[state][1]  # Probability of hitting
            
            # With usable ace
            state = (player_sum, dealer_card, True)
            if state in probs:
                Z_ace[i, j] = probs[state][1]  # Probability of hitting
    
    # Create 3D plots
    fig = plt.figure(figsize=(15, 6))
    
    # No usable ace
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_no_ace, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Dealer Upcard')
    ax1.set_ylabel('Player Sum')
    ax1.set_zlabel('Hit Probability')
    ax1.set_title('Hit Probability: No Usable Ace')
    
    # With usable ace
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_ace, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('Dealer Upcard')
    ax2.set_ylabel('Player Sum')
    ax2.set_zlabel('Hit Probability')
    ax2.set_title('Hit Probability: With Usable Ace')
    
    plt.tight_layout()
    plt.show()

def evaluate_agent(agent, episodes=10000):
    """Evaluate the trained agent"""
    wins = 0
    losses = 0
    ties = 0
    
    for _ in range(episodes):
        state = agent.env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            state, reward, done = agent.env.step(action)
        
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            ties += 1
    
    win_rate = wins / episodes
    print(f"Evaluation over {episodes} games:")
    print(f"Wins: {wins} ({wins/episodes:.3f})")
    print(f"Losses: {losses} ({losses/episodes:.3f})")
    print(f"Ties: {ties} ({ties/episodes:.3f})")
    print(f"Win rate: {win_rate:.3f}")
    
    return win_rate

if __name__ == "__main__":
    print("Training Blackjack Q-Learning Agent...")
    
    # Create and train agent
    agent = BlackjackQLearning(learning_rate=0.1, discount_factor=0.95, epsilon=0.3)
    agent.train(episodes=500000)
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    evaluate_agent(agent)
    
    # Create visualizations
    print("\nCreating strategy chart...")
    create_strategy_chart(agent)
    
    print("Creating probability surface...")
    create_probability_surface(agent)
    
    print("Done! The agent has been trained and visualizations are displayed.")
