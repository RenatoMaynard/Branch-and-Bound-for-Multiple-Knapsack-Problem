import sys
import pandas as pd
from datetime import datetime, timedelta

###############################################################################
#  Function: import_data_file
#  Purpose:  Reads a text file containing multiple parameters for the 
#            multiple-knapsack-with-pairwise-benefits problem. 
#
#  Format expected in each file (first three lines):
#    1st line: I B C T
#       - I = number of items
#       - B = number of backpacks
#       - C = maximum capacity for each backpack
#       - T = maximum runtime (seconds)
#    2nd line: I integers => each item’s weight
#    3rd line: I integers => each item’s individual benefit
#    subsequent lines: the pairwise benefit matrix in some row-based form
#
#  Example:
#    10 2 50 300
#    10 8 4 6 7 2 9 1 12 5
#    10 15 2 10 3 5 7 20 8 9
#    ... (lines for pairwise benefits) ...
###############################################################################
def import_data_file(file_path):
    """
    Reads a .txt file containing:
      - I, B, C, T in the first line
      - the item weights in the second line
      - the item benefits in the third line
      - subsequent lines describing pairwise benefits

    Returns:
      T, I, B, C, 
      benefits_list (1D), 
      weights_list (1D), 
      pairwise_matrix (2D).
    """
    print(f"\n----- Reading instance: {file_path} -----")
    with open(file_path,'r') as f:
        data = f.readlines()

    try:
        # 1) Parse the first line
        header = data[0].split()
        I = int(header[0])  # Number of Items
        B = int(header[1])  # Number of Backpacks
        C = int(header[2])  # Max capacity each backpack
        T = int(header[3])  # Max runtime (seconds)

        # 2) Parse second line => item weights
        weights_line = data[1].split()
        weights_list = [int(w) for w in weights_line]

        # 3) Parse third line => item benefits
        benefits_line = data[2].split()
        benefits_list = [int(b) for b in benefits_line]

        # 4) The rest => pairwise benefit matrix
        matrix_data = data[3:]  # from line 4 onward
        pairwise_matrix = []
        for line in matrix_data:
            row_values = [int(value) for value in line.split()]
            pairwise_matrix.append(row_values)

        print(f"Max Time (seconds):  {T}")
        print(f"Num Items:          {I}")
        print(f"Num Backpacks:      {B}")
        print(f"Capacity/Backpack:  {C}")
        print("Item Benefits:", benefits_list)
        print("Item Weights:", weights_list)
        # Optional: print the entire pairwise matrix
        # for row in pairwise_matrix:
        #     print(row)

    except:
        print("Error: The input file is not formatted correctly. Check your data.")
        sys.exit(1)

    return T, I, B, C, benefits_list, weights_list, pairwise_matrix


###############################################################################
#  Function: create_pair_benefits
#  Purpose:  Converts a 2D pairwise-benefit matrix into a dictionary 
#            for quick lookups: (i, j) -> benefit.
###############################################################################
def create_pair_benefits(benefits_matrix):
    """
    Takes the 2D pairwise matrix and turns it into a dictionary of 
    {(i, j): pairwise_benefit} where i < j. 
    """
    pair_benefits = {}
    try:
        for i in range(len(benefits_matrix)):
            # For row i, the columns might define pairwise with i+someOffset
            # But the user code below was offset-based.
            for j in range(i + 1, len(benefits_matrix) + 1):
                # The original logic was: benefits_matrix[i][j - i - 1]
                # This depends on how the user structured their matrix lines.
                pair_benefits[(i, j)] = benefits_matrix[i][j - i - 1]
    except:
        print("Error building pairwise benefit dictionary. Check matrix format.")

    return pair_benefits


###############################################################################
#  Class: Backpack
#  Purpose:  Represents one backpack, tracking capacity, items, and total benefit
###############################################################################
class Backpack:
    def __init__(self, backpack_id, capacity, current_weight=0):
        self.backpack_id = backpack_id
        self.capacity = capacity
        self.current_weight = current_weight
        self.items = []         # which items are in this backpack
        self.total_benefit = 0  # includes individual benefits + pairwise gains

    def can_add(self, item_weight):
        """Check if there's enough capacity left to add an item."""
        return (self.current_weight + item_weight) <= self.capacity

    def add(self, item_index, weight, benefit, pair_benefits_dict):
        """
        Add an item into the backpack. 
        Also adds any pairwise benefits with items already inside.
        """
        self.current_weight += weight
        self.items.append(item_index)
        self.total_benefit += benefit

        # Add pairwise benefits with previously placed items
        for existing_item in self.items[:-1]:
            # pairwise_benefits stored with (smaller, larger) as the key
            i_sm = min(existing_item, item_index)
            i_bg = max(existing_item, item_index)
            self.total_benefit += pair_benefits_dict.get((i_sm, i_bg), 0)

    def copy(self):
        """Returns a new Backpack object with the same content."""
        bp = Backpack(self.backpack_id, self.capacity, self.current_weight)
        bp.items = self.items.copy()
        bp.total_benefit = self.total_benefit
        return bp

    def __str__(self):
        return (f"Backpack {self.backpack_id}: "
                f"capacity {self.capacity}, "
                f"used {self.current_weight}, "
                f"items {self.items}, "
                f"benefit {self.total_benefit}")


###############################################################################
#  Class: TreeNode 
#  Purpose:  Represents a node in the search tree
###############################################################################
class TreeNode:
    next_id = 1

    def __init__(self, level, backpacks):
        self.node_id = TreeNode.next_id
        TreeNode.next_id += 1

        self.level = level
        # Copy the existing backpacks so that each node
        # has a separate “snapshot” of the solution at that point
        self.backpacks = [bp.copy() for bp in backpacks]
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __str__(self):
        backpacks_str = ", ".join(str(bp) for bp in self.backpacks)
        total_benefit = sum(bp.total_benefit for bp in self.backpacks)
        return (f"[Node {self.node_id} | Level: {self.level} | "
                f"Objective: {total_benefit} | Backpacks: {backpacks_str}]")


###############################################################################
#  Function: build_tree
#  Purpose:  Recursively builds out a (potentially) full search tree for 
#            the multiple-knapsack problem, respecting a time limit 
#            and a best-bound approach.
###############################################################################
def build_tree(max_time_duration, start_time, items, weights, benefits,
               pair_benefits_dict, backpacks, max_levels,
               level=0, parent=None, best_so_far=float('-inf')):
    """
    Builds the search tree:
      - If level == max_levels, we are done (all items have been considered).
      - If we exceed the time limit, we also terminate further branching.
      - We create a "no-add" child node, and also multiple "add" child nodes 
        for each backpack (if feasible).
      - We attempt to use a bounding approach to skip expansions that 
        cannot beat 'best_so_far'.

    Args:
        max_time_duration: A timedelta indicating how long we can search.
        start_time: The datetime when the search started.
        items: List of item indices.
        weights: Weights of each item (parallel to items).
        benefits: Individual benefits of each item.
        pair_benefits_dict: dictionary of pairwise benefits.
        backpacks: the current “solution state” (list of Backpack objects).
        max_levels: total number of items to consider (the depth).
        level: current item index we are deciding on (0..max_levels-1).
        parent: the parent TreeNode in the search tree.
        best_so_far: the best objective known so far (for bounding).
    """
    now = datetime.now()
    elapsed = now - start_time
    if level == max_levels:
        # All items processed
        return

    if elapsed > max_time_duration:
        # We ran out of time
        return

    # 1) Always create a branch for "not taking the item at this level"
    no_item_node = TreeNode(level + 1, backpacks)
    if parent is not None:
        parent.add_child(no_item_node)
    # Recursively explore further items
    build_tree(max_time_duration, start_time, items,
               weights, benefits, pair_benefits_dict,
               no_item_node.backpacks, max_levels,
               level + 1, no_item_node, best_so_far)

    # 2) For each backpack, try "adding" the item at 'level', if feasible
    for i, bp in enumerate(backpacks):
        new_backpacks = [x.copy() for x in backpacks]
        if new_backpacks[i].can_add(weights[level]):
            new_backpacks[i].add(level, weights[level],
                                 benefits[level],
                                 pair_benefits_dict)
            total_now = sum(bk.total_benefit for bk in new_backpacks)

            # Potential bounding: estimate max possible from future items
            # e.g. a naive approach is adding sum of all future item benefits:
            future_benefits = sum(benefits[level+1:])
            if total_now + future_benefits > best_so_far:
                # update bound
                best_so_far = total_now

                # create a new child node
                new_node = TreeNode(level + 1, new_backpacks)
                if parent is not None:
                    parent.add_child(new_node)

                # continue building from there
                build_tree(max_time_duration, start_time, items,
                           weights, benefits, pair_benefits_dict,
                           new_backpacks, max_levels,
                           level + 1, new_node, best_so_far)


###############################################################################
#  Function: find_max_obj_node
#  Purpose:  Walks the entire search tree to find the node with the best
#            (maximum) objective value.
###############################################################################
def find_max_obj_node(node, best_node=None):
    """
    Recursively traverses the search tree from 'node' down. 
    Returns the TreeNode that has the highest total objective across all backpacks.
    """
    current_total = sum(bp.total_benefit for bp in node.backpacks)
    if best_node is None:
        best_node = node
    else:
        best_total = sum(bp.total_benefit for bp in best_node.backpacks)
        if current_total > best_total:
            best_node = node

    # Explore children
    for child in node.children:
        best_node = find_max_obj_node(child, best_node)

    return best_node


###############################################################################
#  Function: print_tree
#  WARNING:  This can produce huge output if the tree is large. 
#  Purpose:  Recursively prints the entire tree (for debugging only).
###############################################################################
def print_tree(node, depth=0):
    print("  " * depth + str(node))
    for child in node.children:
        print_tree(child, depth + 1)


###############################################################################
#  Function: heuristic (greedy)
#  Purpose:  A quick approach that sorts items by ratio(benefit/weight),
#            then tries to place them in the backpacks in that order.
###############################################################################
def heuristic_solution(T, I, B, C, benefits, weights, pairwise_matrix):
    """
    Simple greedy approach: 
       - For each item i, compute ratio = benefits[i]/weights[i] (or fallback if weight=0).
       - Sort items descending by ratio.
       - Try to place each item in the first backpack that can fit it.

    Returns: 
       (heuristic_objective, list_of_backpacks)
    """
    # Compute ratio (benefit / weight)
    ratio_list = []
    for i in range(len(weights)):
        w = weights[i] if weights[i] != 0 else 1
        ratio_list.append(round(benefits[i] / w, 2))

    # Sort item indices by ratio descending
    enumerated_ratios = list(enumerate(ratio_list))
    sorted_ratios = sorted(enumerated_ratios, key=lambda x: x[1], reverse=True)
    item_indices_sorted = [x[0] for x in sorted_ratios]

    # Build the pairwise dict from the matrix
    pair_benefits_dict = create_pair_benefits(pairwise_matrix)

    # Create backpacks
    backpacks = [Backpack(i, C) for i in range(B)]

    # Fill each backpack greedily
    for bp in backpacks:
        # While we still have items left
        while item_indices_sorted:
            candidate_item = item_indices_sorted[0]
            if bp.can_add(weights[candidate_item]):
                bp.add(candidate_item,
                       weights[candidate_item],
                       benefits[candidate_item],
                       pair_benefits_dict)
                # remove item from the list (each item can only be used once)
                item_indices_sorted.pop(0)
            else:
                # if we can't fit it, break and try the next backpack
                break

    # Compute total
    total_benef = sum(bk.total_benefit for bk in backpacks)
    return total_benef, backpacks


###############################################################################
#  Function: write_results
#  Purpose:  Print solution data and also export to a text file
###############################################################################
def write_results(instance_name, best_node_or_comment, best_benefit, elapsed_time):
    """
    Writes results to a text file: <instance_name>_output.txt
    Also prints them on screen.
    """
    print(f"---------- {instance_name} ----------")
    print("Node or approach used:", best_node_or_comment)
    print("Best Benefit Found: ", best_benefit)
    print("Elapsed Time (sec): ", elapsed_time)
    print("--------------------------------------")

    file_out = f"{instance_name}_output.txt"
    with open(file_out, 'w', encoding='utf-8') as f:
        f.write(f"Node/Approach: {best_node_or_comment}\n")
        f.write(f"Benefit: {best_benefit}\n")
        f.write(f"Time (sec): {elapsed_time}\n")


###############################################################################
#  MAIN EXECUTION (Example):
#  Processes multiple instance files, uses branch-and-bound, 
#  compares with a quick heuristic, exports results to "results.xlsx".
###############################################################################
if __name__ == "__main__":

    # Example list of instance text files you want to process:
    instance_files = [
        "instance_A.txt",
        "instance_B.txt",
        "instance_C.txt",
        "instance_D.txt",
        "instance_E.txt",
        "instance_F.txt"
    ]

    
    known_optima = [512, 1745, 3005, 1454, 2011, 650]  

    all_results = []  # will store rows for our final results

    for idx, file_name in enumerate(instance_files):
        try:
            # 1) Read data
            T, I, B, C, benefits, weights, matrix_2D = import_data_file(file_name)

            # 2) Build root solution (empty)
            root_backpacks = [Backpack(i, C) for i in range(B)]
            root_node = TreeNode(0, root_backpacks)

            # 3) Start timing
            start_time = datetime.now()
            max_time = timedelta(seconds=T) if T > 0 else timedelta(seconds=300)

            # 4) Build search tree (branch-and-bound)
            build_tree(max_time, start_time, list(range(I)), weights, benefits,
                       create_pair_benefits(matrix_2D),
                       root_backpacks, I, parent=root_node, best_so_far=0)

            # 5) Get heuristic solution for comparison
            heur_value, heur_bps = heuristic_solution(T, I, B, C, benefits, weights, matrix_2D)

            # 6) Find best node in the generated tree
            best_node = find_max_obj_node(root_node)
            best_node_value = sum(bp.total_benefit for bp in best_node.backpacks)

            # 7) Compare branch-and-bound result vs. heuristic
            #    If the tree result is < heuristic, pick the heuristic result
            if best_node_value < heur_value:
                final_benefit = heur_value
                final_node_comment = "Heuristic"
                final_backpacks = heur_bps
            else:
                final_benefit = best_node_value
                final_node_comment = f"NodeID_{best_node.node_id}"
                final_backpacks = best_node.backpacks

            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()

            # 8) Print and write results
            write_results(file_name, final_node_comment, final_benefit, elapsed)

            # 9) Store row in all_results
            row_data = [file_name,
                        (known_optima[idx] if idx < len(known_optima) else None),
                        final_benefit,
                        None,  # Gap or difference from known optimum
                        round(elapsed, 2)]
            # If we have a known optimum, compute the gap
            if row_data[1] is not None and row_data[1] != 0:
                gap = ((row_data[1] - row_data[2]) / row_data[1]) * 100
                row_data[3] = round(gap, 2)

            all_results.append(row_data)

        except Exception as e:
            print(f"Error in instance '{file_name}': {e}")

    # Once all instances processed, we can optionally create an average row:
    if all_results:
        # Summaries
        sum_opt = 0
        sum_benef = 0
        sum_gap = 0
        sum_time = 0
        count = 0
        for row in all_results:
            # row => [inst_name, known_optimum, found_benefit, gap, time]
            if row[1] is not None:
                sum_opt += row[1]
            sum_benef += row[2]
            if row[3] is not None:
                sum_gap += row[3]
            sum_time += row[4]
            count += 1

        average_opt = sum_opt / count
        average_ben = sum_benef / count
        average_gap = sum_gap / count
        average_time = sum_time / count

        # Build "Average" row
        avg_row = ["Average", average_opt, average_ben, average_gap, average_time]
        all_results.append(avg_row)

        # Convert to dataframe and export
        df = pd.DataFrame(all_results,
                          columns=["Instance", "Optimum", "Found_Benefit", "GAP (%)", "Time (sec)"])
        df.to_excel("results.xlsx", index=False)
        print("\nFinal results (also in results.xlsx):\n", df)
    else:
        print("No instances were processed or no results to display.")
