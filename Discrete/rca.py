import math

# check prime
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

# gcd
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# Compute public exponent e= (1 < e < φ(n), gcd(e, φ(n)) = 1)
def calculate_e(phi_n):
    for e in range(2, phi_n):
        if gcd(e, phi_n) == 1:
            return e
    raise ValueError("Failed to find suitable e")

# Compute private exponent d (d * e ≡ 1 mod φ(n))
def calculate_d(e, phi_n):
    k = 1
    while True:
        k += phi_n
        if k % e == 0:
            return k // e

# Encryption: convert to special characters instead of numeric array
def encrypt_char(char, e, n):
    current = ord(char) - 97  # 'a' -> 0
    result = 1
    for _ in range(e):
        result = (result * current) % n
    # Convert to printable special character range (33–126)
    # ensure not to exceed ASCII printable range
    encoded = chr((result % 94) + 33)
    return encoded

# Decryption: reverse special char -> number -> plaintext
def decrypt_char(enc_char, d, n):
    value = (ord(enc_char) - 33) % 94
    result = 1
    for _ in range(d):
        result = (result * value) % n
    return chr((result + 97) % 256)  # restore ASCII character

# Main program
def main():
    print("Welcome to RSA Program (Special Character Encryption)\n")
    # Step 1: Input prime p
    while True:
        p = int(input("Enter a prime number p: "))
        if is_prime(p):
            break
        print(" WRONG INPUT: Not a prime number.\n")

    # Step 2: Input prime q
    while True:
        q = int(input("Enter a prime number q: "))
        if is_prime(q):
            break
        print(" WRONG INPUT: Not a prime number.\n")

    # Step 3: Compute n = p * q
    n = p * q
    print(f"\nResult of computing n = p * q = {n}")

    # Step 4: Compute φ(n)
    phi_n = (p - 1) * (q - 1)
    print(f"Result of φ(n) = {phi_n}")

    # Step 5: Compute e and d
    e = calculate_e(phi_n)
    d = calculate_d(e, phi_n)

    print(f"\nRSA Public Key (e, n) = ({e}, {n})")
    print(f"RSA Private Key (d, n) = ({d}, {n})")

    # Step 6: Input message
    msg = input("\nEnter message to be encrypted (lowercase letters only): ").lower()
    print(f"\nThe message is: {msg}")

    # Step 7: Encryption
    encrypted_chars = [encrypt_char(ch, e, n) for ch in msg]
    encrypted_msg = ''.join(encrypted_chars)
    print("\nTHE ENCRYPTED MESSAGE IS (special characters):")
    print(encrypted_msg)

    # Step 8: Decryption
    decrypted_chars = [decrypt_char(ch, d, n) for ch in encrypted_chars]
    decrypted_msg = ''.join(decrypted_chars)

    print("\nTHE DECRYPTED MESSAGE IS:")
    print(decrypted_msg)
    print("\nDone.")

if __name__ == "__main__":
    main()
