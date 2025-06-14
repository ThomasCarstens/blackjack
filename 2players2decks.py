import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random
from collections import defaultdict
import pandas as pd

class TwoPlayerBlackjackEnvironment:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Initialize 2-deck shoe (104 cards)
        self.deck = []
        for _ in range(2):  # 2 decks
            for suit in range(4):
                for rank in range(1, 14):  # 1=Ace, 2-10=numbers, 11=J, 12=Q, 13=K
                    self.deck.append(rank)
        random.shuffle(self.deck)
        
        # Deal initial cards to both players and dealer
        self.player1_cards = [self.draw_card(), self.draw_card()]
        self.player2_cards = [self.draw_card(), self.draw_card()]
        self.dealer_cards = [self.draw_card(), self.draw_card()]
        
        self.player1_done = False
        self.player2_done = False
        self.current_player = 1  # Start with player 1
        
        return self.get_state()
    
    def draw_card(self):
        """Draw a card from the deck"""
        if len(self.deck) < 10:  # Reshuffle if deck is low
            self.reset_deck()
        return self.deck.pop()
    
    def reset_deck(self):
        """Reset and shuffle the deck"""
        self.deck = []
        for _ in range(2):  # 2 decks
            for suit in range(4):
                for rank in range(1, 14):
                    self.deck.append(rank)
        random.shuffle(self.deck)
    
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
        """Get current state for both players"""
        p1_sum = self.hand_value(self.player1_cards)
        p1_ace = self.has_usable_ace(self.player1_cards)
        p2_sum = self.hand_value(self.player2_cards)
        p2_ace = self.has_usable_ace(self.player2_cards)
        dealer_upcard = self.card_value(self.dealer_cards[0])
        
        # State includes both players' info and whose turn it is
        return {
            'player1': (p1_sum, dealer_upcard, p1_ace, self.player1_done),
            'player2': (p2_sum, dealer_upcard, p2_ace, self.player2_done),
            'current_player': self.current_player,
            'dealer_upcard': dealer_upcard,
            'cards_remaining': len(self.deck)
        }
    
    def step(self, action):
        """Take action: 0=Stand, 1=Hit"""
        if self.current_player == 1 and not self.player1_done:
            reward = self._player_action(1, action)
            if action == 0 or self.hand_value(self.player1_cards) >= 21:
                self.player1_done = True
                self.current_player = 2
        elif self.current_player == 2 and not self.player2_done:
            reward = self._player_action(2, action)
            if action == 0 or self.hand_value(self.player2_cards) >= 21:
                self.player2_done = True
        
        # If both players are done, dealer plays
        game_over = self.player1_done and self.player2_done
        if game_over:
            final_rewards = self.dealer_play()
            return self.get_state(), final_rewards, True
        
        return self.get_state(), {'player1': 0, 'player2': 0, 'combined': 0}, False
    
    def _player_action(self, player_num, action):
        """Handle individual player action"""
        if action == 1:  # Hit
            if player_num == 1:
                self.player1_cards.append(self.draw_card())
                if self.hand_value(self.player1_cards) > 21:
                    return -1  # Bust
            else:
                self.player2_cards.append(self.draw_card())
                if self.hand_value(self.player2_cards) > 21:
                    return -1  # Bust
        return 0
    
    def dealer_play(self):
        """Dealer plays and determine final rewards"""
        # Dealer hits on soft 17
        while self.hand_value(self.dealer_cards) < 17:
            self.dealer_cards.append(self.draw_card())
        
        dealer_sum = self.hand_value(self.dealer_cards)
        p1_sum = self.hand_value(self.player1_cards)
        p2_sum = self.hand_value(self.player2_cards)
        
        # Calculate individual rewards
        p1_reward = self._calculate_reward(p1_sum, dealer_sum)
        p2_reward = self._calculate_reward(p2_sum, dealer_sum)
        
        return {
            'player1': p1_reward,
            'player2': p2_reward,
            'combined': p1_reward + p2_reward  # Cooperative objective
        }
    
    def _calculate_reward(self, player_sum, dealer_sum):
        """Calculate reward for individual player"""
        if player_sum > 21:  # Player busts
            return -1
        elif dealer_sum > 21:  # Dealer busts
            return 1
        elif player_sum > dealer_sum:
            return 1
        elif player_sum < dealer_sum:
            return -1
        else:
            return 0  # Tie

class CooperativeBlackjackQLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        # Q-table now considers both players' states
        self.q_table = defaultdict(lambda: np.zeros(2))  # 2 actions: stand, hit
        self.env = TwoPlayerBlackjackEnvironment()
    
    def state_to_key(self, state):
        """Convert state dict to hashable key for Q-table"""
        p1_info = state['player1']
        p2_info = state['player2']
        current = state['current_player']
        
        # Include both players' information in the state representation
        return (p1_info[0], p1_info[2], p1_info[3],  # P1: sum, ace, done
                p2_info[0], p2_info[2], p2_info[3],  # P2: sum, ace, done
                state['dealer_upcard'], current)
    
    def get_action(self, state, training=True):
        """Get action using epsilon-greedy policy"""
        state_key = self.state_to_key(state)
        
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            return np.argmax(self.q_table[state_key])
    
    def train(self, episodes=100000):
        """Train the cooperative Q-learning agent"""
        p1_wins = 0
        p2_wins = 0
        combined_wins = 0
        total_combined_reward = 0
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_states = []
            episode_actions = []
            
            while not done:
                current_state = state.copy()
                action = self.get_action(state, training=True)
                next_state, rewards, done = self.env.step(action)
                
                episode_states.append(current_state)
                episode_actions.append(action)
                state = next_state
            
            # Update Q-values based on combined reward (cooperative learning)
            combined_reward = rewards['combined']
            total_combined_reward += combined_reward
            
            # Track individual performance
            if rewards['player1'] > 0:
                p1_wins += 1
            if rewards['player2'] > 0:
                p2_wins += 1
            if combined_reward > 0:
                combined_wins += 1
            
            # Backward pass to update Q-values
            for i in range(len(episode_states)):
                state_key = self.state_to_key(episode_states[i])
                action = episode_actions[i]
                
                # Calculate discounted future reward
                future_reward = 0
                for j in range(i, len(episode_states)):
                    future_reward += (self.gamma ** (j - i)) * combined_reward
                
                # Q-learning update
                current_q = self.q_table[state_key][action]
                self.q_table[state_key][action] += self.lr * (future_reward - current_q)
            
            # Decay epsilon and report progress
            if episode % 10000 == 0 and episode > 0:
                self.epsilon = max(0.01, self.epsilon * 0.95)
                avg_combined = total_combined_reward / episode
                p1_rate = p1_wins / episode
                p2_rate = p2_wins / episode
                combined_rate = combined_wins / episode
                
                print(f"Episode {episode}:")
                print(f"  P1 win rate: {p1_rate:.3f}")
                print(f"  P2 win rate: {p2_rate:.3f}")
                print(f"  Combined positive rate: {combined_rate:.3f}")
                print(f"  Avg combined reward: {avg_combined:.3f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
        
        final_p1_rate = p1_wins / episodes
        final_p2_rate = p2_wins / episodes
        final_combined_rate = combined_wins / episodes
        final_avg_reward = total_combined_reward / episodes
        
        print(f"\nTraining completed:")
        print(f"  Final P1 win rate: {final_p1_rate:.3f}")
        print(f"  Final P2 win rate: {final_p2_rate:.3f}")
        print(f"  Final combined positive rate: {final_combined_rate:.3f}")
        print(f"  Final avg combined reward: {final_avg_reward:.3f}")
    
    def get_policy_for_player(self, player_num):
        """Extract policy for visualization"""
        policy = {}
        for state_key in self.q_table:
            # Extract relevant info for the specific player
            if state_key[7] == player_num:  # current_player index
                p1_sum, p1_ace, p1_done, p2_sum, p2_ace, p2_done, dealer_up, current = state_key
                
                if player_num == 1 and not p1_done:
                    key = (p1_sum, dealer_up, p1_ace, p2_sum, p2_ace)
                elif player_num == 2 and not p2_done:
                    key = (p2_sum, dealer_up, p2_ace, p1_sum, p1_ace)
                else:
                    continue
                
                policy[key] = np.argmax(self.q_table[state_key])
        
        return policy

def create_cooperative_strategy_chart(agent):
    """Create strategy charts for both players"""
    p1_policy = agent.get_policy_for_player(1)
    p2_policy = agent.get_policy_for_player(2)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # For each player, create charts with and without usable ace
    for player_idx, (policy, player_name) in enumerate([(p1_policy, "Player 1"), (p2_policy, "Player 2")]):
        for ace_idx, ace_state in enumerate([False, True]):
            ax = axes[player_idx, ace_idx]
            
            # Create strategy matrix
            strategy = np.zeros((10, 10))  # Player sums 12-21, Dealer 2-11
            
            for (player_sum, dealer_card, usable_ace, other_sum, other_ace) in policy:
                if usable_ace == ace_state and 12 <= player_sum <= 21:
                    dealer_idx = 9 if dealer_card == 11 else dealer_card - 2
                    player_idx_pos = player_sum - 12
                    
                    if 0 <= dealer_idx < 10 and 0 <= player_idx_pos < 10:
                        action = policy[(player_sum, dealer_card, usable_ace, other_sum, other_ace)]
                        strategy[player_idx_pos, dealer_idx] = action
            
            # Create heatmap
            sns.heatmap(strategy, annot=True, fmt='.0f', cmap='RdYlGn', 
                       xticklabels=['2','3','4','5','6','7','8','9','10','A'],
                       yticklabels=range(12, 22), ax=ax, 
                       cbar_kws={'label': 'Action (0=Stand, 1=Hit)'})
            
            ace_text = "With Usable Ace" if ace_state else "No Usable Ace"
            ax.set_title(f'{player_name} Strategy: {ace_text}')
            ax.set_xlabel('Dealer Upcard')
            ax.set_ylabel('Player Sum')
    
    plt.tight_layout()
    plt.show()

def evaluate_cooperative_agent(agent, episodes=10000):
    """Evaluate the trained cooperative agent"""
    p1_wins = 0
    p2_wins = 0
    p1_losses = 0
    p2_losses = 0
    p1_ties = 0
    p2_ties = 0
    total_combined_reward = 0
    
    for _ in range(episodes):
        state = agent.env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            state, rewards, done = agent.env.step(action)
        
        total_combined_reward += rewards['combined']
        
        # Track individual results
        if rewards['player1'] > 0:
            p1_wins += 1
        elif rewards['player1'] < 0:
            p1_losses += 1
        else:
            p1_ties += 1
            
        if rewards['player2'] > 0:
            p2_wins += 1
        elif rewards['player2'] < 0:
            p2_losses += 1
        else:
            p2_ties += 1
    
    print(f"Evaluation over {episodes} games:")
    print(f"Player 1 - Wins: {p1_wins} ({p1_wins/episodes:.3f}), Losses: {p1_losses} ({p1_losses/episodes:.3f}), Ties: {p1_ties} ({p1_ties/episodes:.3f})")
    print(f"Player 2 - Wins: {p2_wins} ({p2_wins/episodes:.3f}), Losses: {p2_losses} ({p2_losses/episodes:.3f}), Ties: {p2_ties} ({p2_ties/episodes:.3f})")
    print(f"Average combined reward: {total_combined_reward/episodes:.3f}")
    print(f"Combined positive outcomes: {sum(1 for i in range(episodes) if total_combined_reward > 0)/episodes:.3f}")
    
    return total_combined_reward / episodes

def create_cooperation_analysis(agent, episodes=1000):
    """Analyze how cooperation affects decision making"""
    # Compare decisions when partner has different hand strengths
    partner_scenarios = [
        ("Strong partner (20)", 20, False),
        ("Weak partner (15)", 15, False),
        ("Busted partner (22)", 22, False),
        ("Soft partner (A,6)", 17, True)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (scenario_name, partner_sum, partner_ace) in enumerate(partner_scenarios):
        ax = axes[idx]
        
        # Create decision matrix for current player based on partner's hand
        decisions = np.zeros((10, 10))  # Player sums 12-21, Dealer 2-11
        
        for player_sum in range(12, 22):
            for dealer_card in range(2, 12):
                # Create state where partner has specific hand
                state = {
                    'player1': (player_sum, dealer_card, False, False),
                    'player2': (partner_sum, dealer_card, partner_ace, True),
                    'current_player': 1,
                    'dealer_upcard': dealer_card,
                    'cards_remaining': 50
                }
                
                action = agent.get_action(state, training=False)
                dealer_idx = 9 if dealer_card == 11 else dealer_card - 2
                player_idx = player_sum - 12
                
                if 0 <= dealer_idx < 10 and 0 <= player_idx < 10:
                    decisions[player_idx, dealer_idx] = action
        
        sns.heatmap(decisions, annot=True, fmt='.0f', cmap='RdYlGn',
                   xticklabels=['2','3','4','5','6','7','8','9','10','A'],
                   yticklabels=range(12, 22), ax=ax,
                   cbar_kws={'label': 'Action (0=Stand, 1=Hit)'})
        ax.set_title(f'Decisions with {scenario_name}')
        ax.set_xlabel('Dealer Upcard')
        ax.set_ylabel('Current Player Sum')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Training Cooperative 2-Player Blackjack Q-Learning Agent...")
    print("Using 2-deck shoe, players optimize combined rewards")
    
    # Create and train cooperative agent
    agent = CooperativeBlackjackQLearning(learning_rate=0.1, discount_factor=0.95, epsilon=0.3)
    agent.train(episodes=500000)
    
    # Evaluate agent
    print("\nEvaluating trained cooperative agent...")
    evaluate_cooperative_agent(agent)
    
    # Create visualizations
    print("\nCreating cooperative strategy charts...")
    create_cooperative_strategy_chart(agent)
    
    print("Creating cooperation analysis...")
    create_cooperation_analysis(agent)
    
    print("Done! The cooperative agent has been trained and visualizations are displayed.")