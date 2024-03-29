# Jigsaw puzzle solver
A semi-automatic jigsaw puzzle solver from a image of the scattered pieces
## Use instructions
Input images must be in a folder named 'images' in the same directory as the program files.
Press 'c' to continue afer adjusting chroma and corners.

  Parameters:
  
    python main.py {Puzzle height} {Puzzle width} {Mode} {Jump size} {Max alternatives difference} {Inpaint kernel} {Max resolution}
    
  **Puzzle height** and **Puzzle width**:
  
    Heigth and width of the puzzle in number of pieces. Doesn't matter the order.
    
  **Mode** (OPTIONAL, default: 0):
  
     0 - Normal mode + reduced output
     1 - Normal mode + complete output
     2 - Normal mode + complete output + resolution animation
     3 - Step mode + complete output (Step mode will ask you if you want to continue searching for a better solution each time one is found).
     Step mode is recommended if the puzzle have more than 30 pieces and you don't want to increase jump size.
     
  **Jump size** (OPTIONAL, default: 1):
  
    If increased, it will try to start with less pieces. It's recommended to increase it if the program takes a long time.
    For example, if Jump size = 3, the algorithm will only try to start with pieces 0, 3, 6, 9,...
    
  **Max alternatives difference** (OPTIONAL, default: 125):
  
    Value that determines the maximum value difference to consider other piece an alternative to that position.
    If decreased, the algorithm will be more greedy. If you increse the value, the program will take longer but it will return better solutions.
    If you increase it too much, it will be difficult for the algorithm to place every piece.
    It is recommended to decrease it if the whole puzzle have a similar tonality.
    
  **Inpaint kernel** (OPTIONAL, default: 7):
  
    Size of the inpaint area. Increase it if noise appears in the extracted borders.
    
  **Max resolution** (OPTIONAL, default: 1000):
  
    Max resolution of the windows shown. Can't be less than 250.
  
