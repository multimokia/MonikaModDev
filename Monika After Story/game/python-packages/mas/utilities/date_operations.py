### date adjusting functions
import datetime


def add_years(initial_date: datetime.datetime, years: int):
    """
    ASSUMES:
        initial_date as datetime
        years as an int

    IN:
        initial_date: the date to add years to
        years : the number of years to add

    RETURNS:
        the date with the years added, if it's feb 29th it goes to mar 1st,
        if feb 29 doesn't exists in the new year
    """
    try:

        # Simply add the years using replace
        return initial_date.replace(year=initial_date.year + years)
    except ValueError:

        # We handle the only exception feb 29
        return initial_date + (datetime.date(initial_date.year + years, 1, 1)
                               - datetime.date(initial_date.year, 1, 1))


def add_months(starting_date, months):
    """
    Takes a datetime object and add a number of months
    Handles the case where the new month doesn't have that day

    IN:
        starting_date - date representing the date to add months to
        months - number of months to add

    OUT:
        datetime.date representing the inputted date with the corresponding months added
    """
    old_month = starting_date.month
    old_year = starting_date.year
    old_day = starting_date.day

    # To handle F29 consistently with add_years, we explicitly manage it
    if months and (months / 12 + old_year) % 4 != 0 and old_month == 2 and old_day == 29:
        old_month = 3
        old_day = 1

    # get the total of months
    total_months = old_month + months

    # get the new month based on date
    new_month = total_months % 12

    # handle december specially
    new_month = 12 if new_month == 0 else new_month

    # get the new year
    new_year = old_year + int(total_months / 12)
    if new_month == 12:
        new_year -= 1

    # Try adding a month, if that doesn't work (there aren't enough days in the month)
    # keep subtracting days till it works.
    date_worked = False
    reduce_days = 0
    while reduce_days <= 3 and not date_worked:
        try:
            new_date = starting_date.replace(year=new_year, month=new_month, day=old_day - reduce_days)
            date_worked = True
        except ValueError:
            reduce_days += 1

    if not date_worked:
        raise ValueError('Adding months failed')

    return new_date