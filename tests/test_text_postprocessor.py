from whisper_key.text_postprocessor import TextPostProcessor


def test_corrections_replace_case_insensitively_with_key_as_written():
    processor = TextPostProcessor(corrections={'CAPEX': ['cap x', 'copics']})
    assert processor.process('The Cap X budget and copics.') == 'The CAPEX budget and CAPEX.'


def test_longest_variant_wins():
    processor = TextPostProcessor(corrections={'amine absorber': ['immune absorber'], 'amine': ['immune']})
    assert processor.process('the immune absorber') == 'the amine absorber'


def test_corrections_do_not_touch_partial_words():
    processor = TextPostProcessor(corrections={'Kube': ['cube']})
    assert processor.process('cubes and a cube') == 'cubes and a Kube'


def test_strip_trailing_period():
    assert TextPostProcessor(strip_trailing_period=True).process('Hello world.') == 'Hello world'


def test_variants_with_symbols_at_word_edges_match():
    processor = TextPostProcessor(corrections={'C++': ['c plus plus'], '.NET': ['dot net'], 'C#': ['c sharp']})
    assert processor.process('I write c plus plus, dot net and C Sharp.') == 'I write C++, .NET and C#.'
    processor = TextPostProcessor(corrections={'Node.js': ['node.js']})
    assert processor.process('node.js!') == 'Node.js!'
