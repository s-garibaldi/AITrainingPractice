from model import NGramModel

if __name__ == "__main__":
    with open("gpt_01/examples/cursor_text_examples.txt") as f:
        examples = f.read().strip().split('\n')
    model = NGramModel(n=2)
    for line in examples:
        model.train(line)
    print("Sample generated text:")
    for _ in range(5):
        print(model.generate()) 