def test_even_odd():
    try:
        assert even_or_odd(2) == "Even"
        print("Test 1: Correct")
    except AssertionError:
        print("Incorrect")

    try:
        assert even_or_odd(1) == "Odd"
        print("Test 2: Correct")
    except AssertionError:
        print("Incorrect")

    try:
        assert even_or_odd(11) == "Even"
        print("Test 2: Correct")
    except AssertionError:
        print("Incorrect")
