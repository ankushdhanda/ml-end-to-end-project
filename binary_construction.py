"""
Binary Construction - Maximize distinct sums from positions with different bits.

Strategy:
- For any split of k zeros and (n-k) ones, we get at most k*(n-k) distinct sums
- To avoid collisions, place zeros at endpoints and ones in the middle
- This creates sums that don't overlap: {1+2, 1+3, ..., n+(n-1)}
"""

def solve(n):
    """
    add more
    Find binary string of length n that maximizes |f(S)|.
    
    Pattern analysis:
    - Position zeros at index 1 and n (endpoints)
    - Position ones at indices 2 to n-1 (middle)
    - This gives 2 zeros and (n-2) ones
    - Maximum distinct sums: 2*(n-2)
    
    Why this works:
    - Zeros at positions {1, n}, ones at {2, 3, ..., n-1}
    - From zero at 1: sums are {1+2, 1+3, ..., 1+(n-1)} = {3, 4, ..., n}
    - From zero at n: sums are {n+2, n+3, ..., n+(n-1)} = {n+2, n+3, ..., 2n-1}
    - No overlap, so we get all (n-2) + (n-2) = 2(n-2) distinct sums
    """
    if n == 2:
        # Special case: only one 0 and one 1
        # f(S) = {3}, size = 1
        return "01"
    else:
        # General case: n >= 3
        # Pattern: "0" + "1"*(n-2) + "0"
        return "0" + "1" * (n - 2) + "0"


def main():
    t = int(input())
    for _ in range(t):
        n = int(input())
        print(solve(n))


if __name__ == "__main__":
    main()
