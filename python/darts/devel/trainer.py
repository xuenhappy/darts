import sys
from .reader import TorchNerSampleReader
from .sover import TSolver
from .model import NerTrainer
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: %s [ner|decide] smaplefile" % sys.argv[0])
        exit(0)
    test_flags = set(['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh'])
    showSample = sys.argv[-1].strip().lower() in test_flags if len(sys.argv) > 3 else False

    if 'ner' == sys.argv[1]:
        ner_sample = TorchNerSampleReader(sys.argv[2], "O,B-_HWORD,I-_HWORD")
        if showSample:
            for widx, lens, winfo in ner_sample:
                winfo = winfo.numpy()
                print("batch____________________**__")
                print(lens)
                for bidx, line in enumerate(widx):
                    print(ner_sample.dts.decode(line.numpy()))
                    labels = []
                    for w in winfo:
                        if w[0] == bidx:
                            labels.append(w[-1])
                    print(labels)

            exit(0)

        ner_model = NerTrainer(ner_sample.wordsize(), 64, 128, 84, ner_sample.labelsize())
        sover = TSolver(ner_model, ner_sample, {"model_outdir": 'model_bin', 'epoch_num': 6})
        sover.solve()
        exit(0)

    raise Exception("not support mode=%s" % sys.argv[1])
