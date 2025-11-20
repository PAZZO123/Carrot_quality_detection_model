# caesar_guess.py
# Simple interactive Caesar-cipher guessing tool
# Author: (yours)
# Usage: python caesar_guess.py

import string

ALPHABET_LOWER = string.ascii_lowercase
ALPHABET_UPPER = string.ascii_uppercase

def caesar_decrypt(ciphertext: str, key: int) -> str:
    """Decrypts ciphertext using a Caesar cipher with the given key (1-26)."""
    if not (1 <= key <= 26):
        raise ValueError("Key must be between 1 and 26")

    result_chars = []
    for ch in ciphertext:
        if ch.islower():
            idx = ALPHABET_LOWER.index(ch)
            new_idx = (idx - key) % 26
            result_chars.append(ALPHABET_LOWER[new_idx])
        elif ch.isupper():
            idx = ALPHABET_UPPER.index(ch)
            new_idx = (idx - key) % 26
            result_chars.append(ALPHABET_UPPER[new_idx])
        else:
            result_chars.append(ch)  # leave digits, punctuation, spaces unchanged

    return "".join(result_chars)

def show_all_shifts(ciphertext: str):
    """Print all 26 possible shifts for quick brute-force inspection."""
    print("\nAll possible decryptions (key -> plaintext):")
    for k in range(1, 27):
        print(f"{k:2d}: {caesar_decrypt(ciphertext, k)}")
    print()

def interactive_loop():
    print("=== Caesar Cipher Key-Guess Tool ===")
    ciphertext = input("Enter the ciphertext: ").rstrip("\n")
    if not ciphertext:
        print("No ciphertext entered. Exiting.")
        return

    print("\nOptions:")
    print("  [g] Guess a key (1-26)")
    print("  [a] Show all possible decryptions (brute-force)")
    print("  [q] Quit\n")

    while True:
        choice = input("Choose an option (g/a/q): ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            break
        elif choice == "a":
            show_all_shifts(ciphertext)
        elif choice == "g":
            while True:
                key_str = input("Enter your key guess (1-26) or 'b' to go back: ").strip().lower()
                if key_str == "b":
                    break
                if not key_str.isdigit():
                    print("Please enter a number between 1 and 26, or 'b' to go back.")
                    continue
                key = int(key_str)
                if not (1 <= key <= 26):
                    print("Key must be between 1 and 26.")
                    continue
                plaintext = caesar_decrypt(ciphertext, key)
                print(f"\nKey {key} -> {plaintext}\n")
                correct = input("Is this the correct plaintext? (y/n): ").strip().lower()
                if correct == "y":
                    print("Great — key accepted. Exiting.")
                    return
                else:
                    print("Try another key or type 'b' to go back to options.")
        else:
            print("Invalid choice. Pick 'g', 'a', or 'q'.")

if __name__ == "__main__":
    interactive_loop()
#Jusxdebewo uleblui gkysabo, qdt ie ckij ekh cetubi. Juijydw myjx...5. Jxyi iqcfbu mybb ulqbkqju oekh tushofjyed hekjydu jxehekwxbo.