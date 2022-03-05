# Comparar oración a oración. Dejo por las dudas
"""
for a_sent in outputs[6].sents:
    a = str(a_sent).lower()

    for b_sent in targets[6].sents:
        b = str(b_sent).lower()
        print('Oración a: ' + str(a))
        print('Oración b: ' + str(b) + '\n')
        print('Comparación: ')
        seq = dl.SequenceMatcher(None, a.lower(), b.lower())
        d = seq.ratio() * 100
        print(str(d) + '\n')

        print('Matcheos: \n')

        matches = dl.SequenceMatcher(None, a, b).get_matching_blocks()
        for match in matches:
            print(a[match.a:match.a + match.size])
"""