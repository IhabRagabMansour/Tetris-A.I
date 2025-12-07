from settings import *
from njit_startup import get_states_fast
import numpy as np

class FeaturePanel:
    """Display Linear NN feature values in a side panel"""

    def __init__(self):
        # Create a surface below the scoreboard
        self.surface = pygame.Surface((SIDEBAR_WIDTH, GAME_HEIGHT * 0.5))
        self.rect = self.surface.get_rect(topright=(WINDOW_WIDTH - PADDING, WINDOW_HEIGHT - PADDING - int(GAME_HEIGHT * 0.5)))
        self.display_surface = pygame.display.get_surface()

        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 28)

        # Feature values
        self.total_heights = 0
        self.bumpiness = 0
        self.lines_removed = 0
        self.holes = 0
        self.y_pos = 0
        self.pillar = 0

    def update_from_game(self, game):
        """Extract and update feature values from game state"""
        board = (game.board != 0).astype(int)
        cols, self.total_heights, self.bumpiness = get_states_fast(board)

        # Calculate holes
        self.holes = np.sum((board == 0) & (np.cumsum(board != 0, axis=0) > 0))

        # Get y_pos (max height of current piece)
        if game.tetromino and hasattr(game.tetromino, 'blocks'):
            self.y_pos = max(block.pos.y for block in game.tetromino.blocks) if game.tetromino.blocks else 0
        else:
            self.y_pos = 0

        # Check for pillars
        pillar = False
        for i in range(1, len(cols)-1):
            if cols[i-1] - cols[i] >= 3 and cols[i+1] - cols[i] >= 3:
                pillar = True
                break
        if not pillar and len(cols) > 1:
            if cols[1] - cols[0] >= 3 or cols[-2] - cols[-1] >= 3:
                pillar = True

        self.pillar = 1 if pillar else 0
        self.lines_removed = game.lines_removed

    def display_text(self, pos, text, font=None, color='white'):
        """Render text at the specified position"""
        if font is None:
            font = self.font
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(topleft=pos)
        self.surface.blit(text_surface, text_rect)

    def run(self, game):
        """Update and draw the feature panel"""
        self.surface.fill(GRAY)
        self.update_from_game(game)

        # Title
        title = "Linear NN Features"
        self.display_text((10, 10), title, self.title_font, CYAN)

        # Draw separator line
        pygame.draw.line(self.surface, LINE_COLOR, (10, 40), (SIDEBAR_WIDTH - 10, 40), 2)

        # Feature descriptions and values
        features = [
            ("Total Heights", f"{int(self.total_heights)}", "Sum of all column heights"),
            ("Bumpiness", f"{int(self.bumpiness)}", "Height variation between columns"),
            ("Lines Cleared", f"{int(self.lines_removed)}", "Lines cleared this move"),
            ("Holes", f"{int(self.holes)}", "Empty cells under blocks"),
            ("Max Height", f"{int(self.y_pos)}", "Current piece height"),
            ("Pillar", f"{int(self.pillar)}", "Dangerous column spike")
        ]

        y_offset = 55
        line_height = 65

        for i, (label, value, description) in enumerate(features):
            y = y_offset + i * line_height

            # Feature name and value
            color = self._get_feature_color(label, value)
            self.display_text((15, y), f"{label}:", color=color)

            # Value (larger font)
            value_font = pygame.font.Font(None, 32)
            self.display_text((15, y + 20), value, value_font, 'white')

            # Description (smaller, gray)
            desc_font = pygame.font.Font(None, 16)
            self.display_text((15, y + 45), description, desc_font, (150, 150, 150))

        # Display the panel
        self.display_surface.blit(self.surface, self.rect)
        pygame.draw.rect(self.display_surface, LINE_COLOR, self.rect, 2, 2)

    def _get_feature_color(self, label, value):
        """Color code features based on their values (red=bad, green=good, yellow=neutral)"""
        try:
            val = int(value)
        except:
            return 'white'

        if label == "Holes":
            return (255, 100, 100) if val > 3 else (100, 255, 100)
        elif label == "Pillar":
            return (255, 50, 50) if val == 1 else (100, 255, 100)
        elif label == "Bumpiness":
            return (255, 100, 100) if val > 10 else (100, 255, 100) if val < 5 else (255, 255, 100)
        elif label == "Total Heights":
            return (255, 100, 100) if val > 100 else (100, 255, 100) if val < 50 else (255, 255, 100)
        else:
            return (200, 200, 255)
