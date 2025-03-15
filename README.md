# 🎒 Branch and Bound for Multiple Knapsack Problem with Pairwise Benefits

This repository contains a Python implementation of a **branch-and-bound** approach (plus a simple greedy heuristic) to solve a variation of the multiple knapsack problem where items have both individual and pairwise benefits.

## 🚀Features

- **🧳Multiple knapsacks** – each with a fixed capacity.
- **🤝Pairwise benefits** – placing items (i,j) together yields an extra benefit.
- **🌲Branch-and-bound** – systematically builds a search tree, with a simple bounding approach.
- **⚙️Greedy heuristic** – as a fallback or initial solution (sort items by benefit/weight).
- **⏱️Time limit** – automatically stops branching after the user-defined time in each instance.

##📄File Formats

Each instance file must have:

1. **First line**: `I B C T`  
   - `I`: number of items  
   - `B`: number of backpacks  
   - `C`: capacity per backpack  
   - `T`: max time (seconds) for the search  
2. **Second line**: list of `I` item weights.  
3. **Third line**: list of `I` item benefits.  
4. **Subsequent lines**: the pairwise-benefits matrix, typically forming a triangular or row-based structure.  


##🛠️Usage

1. **Clone or Download** the repository:
   ```bash
   git clone https://github.com/RenatoMaynard/multi-knapsack-b&b.py.git

##🤝Contributing
Feel free to open issues or create pull requests:
  1. Fork the repository.
  2. Create a new branch.
  3. Commit changes.
  4. Push to the branch.
  5. Create a Pull Request.

## License
This project is open-sourced under the MIT license.
