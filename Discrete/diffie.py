import math

#  check if a number is prime
def is_prime(p):
    if p < 2:
        return False
    for i in range(2, int(math.sqrt(p)) + 1):
        if p % i == 0:
            return False
    return True
#  find primitive root modulo p
def find_primitive_root(p):
    if p == 2:
        return 1
    phi = p - 1
    factors = set()
    n = phi
    i = 2
    while i * i <= n:
        if n % i == 0:
            factors.add(i)
            while n % i == 0:
                n //= i
        i += 1
    if n > 1:
        factors.add(n)
    # test candidates for primitive root
    for g in range(2, p):
        flag = False
        for factor in factors:
            if pow(g, phi // factor, p) == 1:
                flag = True
                break
        if not flag:
            return g
    return None

# Caesar cipher encryption using key
def encrypt_message(message, key):
    encrypted = ""
    for char in message:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            encrypted += chr((ord(char) - base + key) % 26 + base)
        else:
            encrypted += char
    return encrypted

# Caesar cipher decryption using key
def decrypt_message(ciphertext, key):
    decrypted = ""
    for char in ciphertext:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            decrypted += chr((ord(char) - base - key) % 26 + base)
        else:
            decrypted += char
    return decrypted

# Main program
def main():
    print("===============================================")
    print("   DIFFIE–HELLMAN KEY EXCHANGE SIMULATION")
    print("===============================================\n")

    # Step 1: agree on a prime number p and a generator g
    while True:
        p = int(input("Enter a prime number p: "))
        if is_prime(p):
            break
        print(" Not a prime number. Try again.\n")

    g = find_primitive_root(p)
    print(f" Found primitive root g = {g} (mod {p})\n")

    print("Publicly known values:")
    print(f"    Prime (p) = {p}")
    print(f"    Generator (g) = {g}")
    print("-----------------------------------------------")

    # Step 2: Alice chooses private key a
    a = int(input("Enter Alice’s private key (a): "))
    A = pow(g, a, p)
    print(f"Alice computes public key A = g^a mod p = {A}")

    # Step 3: Bob chooses private key b
    b = int(input("Enter Bob’s private key (b): "))
    B = pow(g, b, p)
    print(f"Bob computes public key B = g^b mod p = {B}")

    print("\nValues exchanged publicly over the network:")
    print(f"    Alice → Bob: A = {A}")
    print(f"    Bob → Alice: B = {B}")

    print("\n-----------------------------------------------")

    # Step 4: Each computes shared secret
    secret_Alice = pow(B, a, p)
    secret_Bob = pow(A, b, p)

    print("Both compute shared secret:")
    print(f"    Alice computes: K = (B^a) mod p = {secret_Alice}")
    print(f"    Bob computes:   K = (A^b) mod p = {secret_Bob}")

    if secret_Alice == secret_Bob:
        shared_key = secret_Alice
        print(f"\n Shared symmetric key established: {shared_key}")
    else:
        print("\n Error: Keys do not match!")
        return

    print("\n-----------------------------------------------")

    # Step 5: Message encryption using shared key
    message = input("Enter a message to encrypt: ")

    # To keep encryption manageable, reduce the key to a value between 1–25
    encryption_key = shared_key % 26
    if encryption_key == 0:
        encryption_key = 3  # avoid zero shift

    encrypted = encrypt_message(message, encryption_key)
    decrypted = decrypt_message(encrypted, encryption_key)

    print(f"\nEncryption key (based on shared secret): {encryption_key}")
    print(f"Encrypted message: {encrypted}")
    print(f"Decrypted message: {decrypted}")

    print("\n-----------------------------------------------")
    print("Diffie–Hellman secure communication simulation completed!")

if __name__ == "__main__":
    main()
