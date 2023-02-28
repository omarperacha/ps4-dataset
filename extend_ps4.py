import sys
import ps4_rs

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            ps4_rs.parse_new_samples(sys.argv[1], sys.argv[2])
        else:
            ps4_rs.parse_new_samples(sys.argv[1], 'ps4_data/data/ps4_extended.csv')
    else:
        print('Please specify a path to a directory containing DSSP files')

