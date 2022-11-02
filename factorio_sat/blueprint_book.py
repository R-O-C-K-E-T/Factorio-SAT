import argparse
import copy

from . import blueprint


def unpack_book(blueprint_book):
    return blueprint_book['blueprint_book']['blueprints']


BLUEPRINT_BOOK_TEMPLATE = {'blueprint_book': {'item': 'blueprint-book', 'active_index': 0, 'version': 281474976710656}}


def pack_book(blueprints, label=None):
    blueprint_book = copy.deepcopy(BLUEPRINT_BOOK_TEMPLATE)
    if label is not None:
        blueprint_book['blueprint_book']['label'] = label

    blueprints = list(copy.deepcopy(blueprints))

    for i, item in enumerate(blueprints):
        item['index'] = i

    blueprint_book['blueprint_book']['blueprints'] = blueprints
    return blueprint_book


def main():
    parser = argparse.ArgumentParser(description='Manipulates blueprint books')
    parser.add_argument('mode', choices=['pack', 'unpack'])
    parser.add_argument('--label', type=str, help='Output blueprint book label')
    args = parser.parse_args()

    if args.mode == 'unpack':
        data = blueprint.decode_blueprint(input())
        blueprints = unpack_book(data)

        for item in blueprints:
            print(blueprint.encode_blueprint(item))
    else:
        blueprints = []
        while True:
            try:
                blueprints.append(blueprint.decode_blueprint(input()))
            except EOFError:
                break
        print(blueprint.encode_blueprint(pack_book(blueprints, args.label)))


if __name__ == '__main__':
    main()
