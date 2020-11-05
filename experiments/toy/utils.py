from prettytable import PrettyTable


def get_args_table(args_dict):
    """
    Get table args_dict.

    Args:
        args_dict: (dict): write your description
    """
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table
