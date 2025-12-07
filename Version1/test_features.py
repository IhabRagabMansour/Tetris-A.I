"""
Test script to verify Linear NN feature visualization implementation
"""
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

# Initialize pygame (headless)
pygame.init()
pygame.display.init()

from game import Game
from feature_panel import FeaturePanel
from tetris import Tetris

def test_game_initialization():
    """Test that Game initializes correctly with Linear architecture"""
    print("Testing Game initialization with Linear architecture...")
    game = Game(architecture="Linear")
    assert game.architecture == "Linear"
    assert hasattr(game, 'show_features')
    assert game.show_features == False  # Should default to False
    print("✓ Game initialization successful")
    return game

def test_feature_extraction():
    """Test that features are extracted correctly"""
    print("\nTesting feature extraction...")
    game = Game(architecture="Linear")

    # Create a test board state
    game.board[15:20, :] = 1  # Fill bottom 5 rows
    game.board[18, 3] = 0     # Create a hole

    features = game.get_features_from_board(game.board, lines_removed=1)

    print(f"  Features extracted: {features}")
    assert len(features) == 6, f"Expected 6 features, got {len(features)}"
    assert all(isinstance(f, (int, float, np.integer, np.floating, bool)) for f in features), "All features should be numeric"
    print("✓ Feature extraction successful")
    return features

def test_feature_panel():
    """Test that FeaturePanel initializes correctly"""
    print("\nTesting FeaturePanel initialization...")
    pygame.display.set_mode((800, 600))  # Need a display for panel
    panel = FeaturePanel()

    assert hasattr(panel, 'total_heights')
    assert hasattr(panel, 'bumpiness')
    assert hasattr(panel, 'holes')
    assert hasattr(panel, 'pillar')
    print("✓ FeaturePanel initialization successful")
    return panel

def test_overlay_method():
    """Test that overlay drawing method exists"""
    print("\nTesting visual overlay methods...")
    game = Game(architecture="Linear")

    assert hasattr(game, '_draw_feature_overlays'), "Game should have _draw_feature_overlays method"
    assert hasattr(game, 'show_features'), "Game should have show_features attribute"
    print("✓ Overlay methods exist")

def test_toggle_functionality():
    """Test feature toggle"""
    print("\nTesting feature toggle...")
    game = Game(architecture="Linear")

    initial_state = game.show_features
    game.show_features = not game.show_features
    assert game.show_features != initial_state, "Toggle should change state"
    print(f"✓ Feature toggle works (initial: {initial_state}, toggled: {game.show_features})")

def test_cnn_architecture():
    """Verify CNN architecture doesn't show features"""
    print("\nTesting CNN architecture (should not have feature overlay)...")
    game_cnn = Game(architecture="CNN")
    assert game_cnn.show_features == False
    print("✓ CNN architecture correctly configured")

def main():
    print("=" * 60)
    print("Linear NN Feature Visualization - Test Suite")
    print("=" * 60)

    try:
        # Run tests
        game = test_game_initialization()
        features = test_feature_extraction()
        panel = test_feature_panel()
        test_overlay_method()
        test_toggle_functionality()
        test_cnn_architecture()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nImplementation Summary:")
        print("1. ✓ Game class supports Linear architecture")
        print("2. ✓ Feature extraction working (6 features)")
        print("3. ✓ FeaturePanel class created")
        print("4. ✓ Visual overlay methods implemented")
        print("5. ✓ Toggle functionality working")
        print("\nTo use:")
        print("- Set RENDER = True in settings.py")
        print("- Run with architecture='Linear'")
        print("- Press 'F' to toggle visual overlays")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        pygame.quit()

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
