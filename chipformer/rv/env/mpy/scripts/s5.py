def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True

words = "the quick brown fox jumps over the lazy dog".split()
counts = {}
for w in words:
    counts[w] = counts.get(w, 0) + 1
primes = [n for n in range(50) if is_prime(n)]
print(primes)
print(sorted(counts.items()))
print(max(words, key=len))
