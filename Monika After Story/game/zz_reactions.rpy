# FileReactions framework.
# not too different from events

default persistent._mas_filereacts_failed_map = dict()
# mapping of failed deleted file reacts

default persistent._mas_filereacts_just_reacted = False
# True if we just reacted to something

default persistent._mas_filereacts_reacted_map = dict()
# mapping of file reacts that we have already reacted to today

default persistent._mas_filereacts_stop_map = dict()
# mapping of file reacts that we should no longer react to ever again

default persistent._mas_filereacts_historic = dict()
# historic database used to track when and how many gifts Monika has received

default persistent._mas_filereacts_last_reacted_date = None
# stores the last date gifts were received so we can clear _mas_filereacts_reacted_map

init 800 python:
    if len(persistent._mas_filereacts_failed_map) > 0:
        store.mas_filereacts.delete_all(persistent._mas_filereacts_failed_map)

init -1 python in mas_filereacts:
    import store
    import store.mas_utils as mas_utils
    import datetime
    import random

    # file react database
    filereact_db = dict()

    # file reaction filename mapping
    # key: filename or list of filenames
    # value: Event
    filereact_map = dict()

    # currently found files react map
    # NOTE: highly volitatle. Expect this to change often
    # key: lowercase filename, without extension
    # value: on disk filename
    foundreact_map = dict()

    # spare foundreact map, designed for threaded use
    # same keys/values as foundreact_map
    th_foundreact_map = dict()

    # good gifts list
    good_gifts = list()

    # bad gifts list
    bad_gifts = list()

    # connector quips
    connectors = None
    gift_connectors = None

    # starter quips
    starters = None
    gift_starters = None

    def addReaction(ev_label, fname, _action=store.EV_ACT_QUEUE, is_good=None):
        """
        Adds a reaction to the file reactions database.

        IN:
            ev_label - label of this event
            fname - filename to react to
            _action - the EV_ACT to do
                (Default: EV_ACT_QUEUE)
            is_good - if the gift is good(True), neutral(None) or bad(False)
               (Default: None)
        """
        # lowercase the list in case
        if fname is not None:
            fname = fname.lower()


        # build new Event object
        ev = store.Event(
            store.persistent.event_database,
            ev_label,
            category=fname,
            action=_action
        )

        # add it to the db and map
        filereact_db[ev_label] = ev
        filereact_map[fname] = ev

        if is_good is not None:
            if is_good:
                good_gifts.append(ev_label)
            else:
                bad_gifts.append(ev_label)


    def _initConnectorQuips():
        """
        Initializes the connector quips
        """
        global connectors, gift_connectors

        # the connector is a MASQipList
        connectors = store.MASQuipList(allow_glitch=False, allow_line=False)
        gift_connectors = store.MASQuipList(allow_glitch=False, allow_line=False)


    def _initStarterQuips():
        """
        Initializes the starter quips
        """
        global starters, gift_starters

        # the starter is a MASQuipList
        starters = store.MASQuipList(allow_glitch=False, allow_line=False)
        gift_starters = store.MASQuipList(allow_glitch=False, allow_line=False)


    def react_to_gifts(found_map, connect=True):
        """
        call this function when you want to check files for reacting to gifts.

        IN:
            found_map - dict to use to insert found items.
                NOTE: this function does NOT empty this dict.
            connect - True will add connectors in between each reaction label
                (Default: True)

        RETURNS:
            list of event labels in the order they should be shown
        """

        GIFT_EXT = ".gift"
        raw_gifts = store.mas_docking_station.getPackageList(GIFT_EXT)

        if len(raw_gifts) == 0:
            return []

        # is it a new day?
        if store.persistent._mas_filereacts_last_reacted_date is None or store.persistent._mas_filereacts_last_reacted_date != datetime.date.today():
            store.persistent._mas_filereacts_last_reacted_date = datetime.date.today()
            store.persistent._mas_filereacts_reacted_map = dict()

        # otherwise we found some potential gifts
        gifts_found = list()
        # now lets lowercase this list whie also buliding a map of files
        for _gift in raw_gifts:
            gift_name, ext, garbage = _gift.partition(GIFT_EXT)
            c_gift_name = gift_name.lower()
            if (
                    c_gift_name not in
                        store.persistent._mas_filereacts_failed_map
                    and c_gift_name not in
                        store.persistent._mas_filereacts_reacted_map
                    and c_gift_name not in
                        store.persistent._mas_filereacts_stop_map
                ):
                gifts_found.append(c_gift_name)
                found_map[c_gift_name] = _gift
                store.persistent._mas_filereacts_reacted_map[c_gift_name] = _gift

        # then sort the list
        gifts_found.sort()

        # now we are ready to check for reactions
        # first we check for all file reacts:
        #all_reaction = filereact_map.get(gifts_found, None)

        #if all_reaction is not None:
        #    return [all_reaction.eventlabel]

        # otherwise, we need to do this more carefully
        found_reacts = list()
        for index in range(len(gifts_found)-1, -1, -1):
            _gift = gifts_found[index]
            reaction = filereact_map.get(_gift, None)

            if _gift is not None and reaction is not None:
                # remove from the list and add to found
                # TODO add to the persistent react map today
                gifts_found.pop()
                found_reacts.append(reaction.eventlabel)
                found_reacts.append(gift_connectors.quip()[1])

        # add in the generic gift reactions
        generic_reacts = list()
        if len(gifts_found) > 0:
            for _gift in gifts_found:
                generic_reacts.append("mas_reaction_gift_generic")
                generic_reacts.append(gift_connectors.quip()[1])
                # keep stats for today
                _register_received_gift("mas_reaction_gift_generic")


        generic_reacts.extend(found_reacts)

        # gotta remove the extra
        if len(generic_reacts) > 0:
            generic_reacts.pop()

            # add the ender
            generic_reacts.insert(0, "mas_reaction_end")

            # add the starter
            if store.mas_isMonikaBirthday():
                generic_reacts.append("mas_reaction_gift_starter_bday")
            else:
                generic_reacts.append("mas_reaction_gift_starter_neutral")
#            generic_reacts.append(gift_starters.quip()[1])

        # now return the list
        return generic_reacts


    def _core_delete(_filename, _map):
        """
        Core deletion file function.

        IN:
            _filename - name of file to delete, if None, we delete one randomly
            _map - the map to use when deleting file.
        """
        if len(_map) == 0:
            return

        # otherwise check for random deletion
        if _filename is None:
            _filename = random.choice(_map.keys())

        file_to_delete = _map.get(_filename, None)
        if file_to_delete is None:
            return

        if store.mas_docking_station.destroyPackage(file_to_delete):
            # file has been deleted (or is gone). pop and go
            _map.pop(_filename)
            return

        # otherwise add to the failed map
        store.persistent._mas_filereacts_failed_map[_filename] = file_to_delete


    def _core_delete_list(_filename_list, _map):
        """
        Core deletion filename list function

        IN:
            _filename - list of filenames to delete.
            _map - the map to use when deleting files
        """
        for _fn in _filename_list:
            _core_delete(_fn, _map)


    def _register_received_gift(eventlabel):
        """
        Registers when player gave a gift successfully
        IN:
            eventlabel - the event label for the gift reaction

        """
        # check for stats dict for today
        today = datetime.date.today()
        if not today in store.persistent._mas_filereacts_historic:
            store.persistent._mas_filereacts_historic[today] = dict()

        # Add stats
        store.persistent._mas_filereacts_historic[today][eventlabel] = store.persistent._mas_filereacts_historic[today].get(eventlabel,0) + 1


    def _get_full_stats_for_date(date=None):
        """
        Getter for the full stats dict for gifts on a given date
        IN:
            date - the date to get the report for, if None is given will check
                today's date
                (Defaults to None)

        RETURNS:
            The dict containing the full stats or None if it's empty

        """
        if date is None:
            date = datetime.date.today()
        return store.persistent._mas_filereacts_historic.get(date,None)


    def delete_file(_filename):
        """
        Deletes a file off the found_react map

        IN:
            _filename - the name of the file to delete. If None, we delete
                one randomly
        """
        _core_delete(_filename, foundreact_map)


    def delete_files(_filename_list):
        """
        Deletes multiple files off the found_react map

        IN:
            _filename_list - list of filenames to delete.
        """
        for _fn in _filename_list:
            delete_file(_fn)


    def th_delete_file(_filename):
        """
        Deletes a file off the threaded found_react map

        IN:
            _filename - the name of the file to delete. If None, we delete one
                randomly
        """
        _core_delete(_filename, th_foundreact_map)


    def th_delete_files(_filename_list):
        """
        Deletes multiple files off the threaded foundreact map

        IN:
            _filename_list - list of ilenames to delete
        """
        for _fn in _filename_list:
            th_delete_file(_fn)


    def delete_all(_map):
        """
        Attempts to delete all files in the given map.
        Removes files in that map if they dont exist no more

        IN:
            _map - map to delete all
        """
        _map_keys = _map.keys()
        for _key in _map_keys:
            _core_delete(_key, _map)

    def get_report_for_date(date=None):
        """
        Generates a report for all the gifts given on the input date.
        The report is in tuple form (total, good_gifts, neutral_gifts, bad_gifts)
        it contains the totals of each type of gift.
        """
        if date is None:
            date = datetime.date.today()

        stats = _get_full_stats_for_date(date)
        if stats is None:
            return (0,0,0,0)
        good = 0
        bad = 0
        neutral = 0
        for _key in stats.keys():
            if _key in good_gifts:
                good = good + stats[_key]
            if _key in bad_gifts:
                bad = bad + stats[_key]
            if _key == "":
                neutral = stats[_key]
        total = good + neutral + bad
        return (total, good, neutral, bad)



    # init
    _initConnectorQuips()
    _initStarterQuips()

init python:
    import store.mas_filereacts as mas_filereacts

    def addReaction(ev_label, fname_list, _action=EV_ACT_QUEUE, is_good=None):
        """
        Globalied version of the addReaction function in the mas_filereacts
        store.

        Refer to that function for more information
        """
        mas_filereacts.addReaction(ev_label, fname_list, _action, is_good)


    def mas_checkReactions():
        """
        Checks for reactions, then queues them
        """
        # only check if we didnt just react
        if persistent._mas_filereacts_just_reacted:
            return

        # otherwise check
        mas_filereacts.foundreact_map.clear()
        reacts = mas_filereacts.react_to_gifts(mas_filereacts.foundreact_map)
        if len(reacts) > 0:
            # need to reverse it now
            reacts.reverse()
            for _react in reacts:
                queueEvent(_react)
            persistent._mas_filereacts_just_reacted = True


    def mas_receivedGift(ev_label):
        """
        Globalied version for gift stats tracking
        """
        mas_filereacts._register_received_gift(ev_label)


    def mas_generateGiftsReport(date=None):
        """
        Globalied version for gift stats tracking
        """
        return mas_filereacts.get_report_for_date(date)

    def mas_getGiftStatsForDate(label,date=None):
        """
        Globalied version to get the stats for a specific gift
        IN:
            label - the gift label identifier.
            date - the date to get the stats for, if None is given will check
                today's date.
                (Defaults to None)

        RETURNS:
            The number of times the gift has been given that date
        """
        if date is None:
            date = datetime.date.today()
        historic = persistent._mas_filereacts_historic.get(date,None)

        if historic is None:
            return 0
        return historic.get(label,0)



### CONNECTORS [RCT000]

# none here!

## Gift CONNECTORS [RCT010]
#
#init 5 python:
#    store.mas_filereacts.gift_connectors.addLabelQuip(
#        "mas_reaction_gift_connector_test"
#    )

label mas_reaction_gift_connector_test:
    m "this is a test of the connector system"
    return

init 5 python:
    store.mas_filereacts.gift_connectors.addLabelQuip(
        "mas_reaction_gift_connector1"
    )

label mas_reaction_gift_connector1:
    m 1sublo "Oh! There was something else you wanted to give me?"
    m 1hua "Well! I better open it quickly, shouldn't I?"
    m 1suo "And here we have..."
    return

init 5 python:
    store.mas_filereacts.gift_connectors.addLabelQuip(
        "mas_reaction_gift_connector2"
    )

label mas_reaction_gift_connector2:
    m 1hua "Ah, jeez, [player]..."
    m "You really enjoy spoiling me, don't you?"
    if mas_isSpecialDay():
        m 1sublo "Well! I'm not going to complain about a little special treatment today."
    m 1suo "And here we have..."
    return


### STARTERS [RCT050]

init 5 python:
    store.mas_filereacts.gift_starters.addLabelQuip(
        "mas_reaction_gift_starter_generic"
    )

label mas_reaction_gift_starter_generic:
    m "generic test"

# init 5 python:
# TODO: if we need this to be multipled then we do it

label mas_reaction_gift_starter_bday:
    m 1sublo ". {w=0.7}. {w=0.7}. {w=1}"
    m "T-{w=1}This is..."
    m "A gift? For me?"
    m 1hka "I..."
    m 1hua "I've often thought about getting presents from you on my birthday..."
    m "But actually getting one is like a dream come true..."
    m 1sua "Now, what's inside?"
    m 1suo "Oh, it's..."
    return

label mas_reaction_gift_starter_neutral:
    m 1sublo ". {w=0.7}. {w=0.7}. {w=1}"
    m "T-{w=1}This is..."
    m "A gift? For me?"
    m 1sua "Now, let's see what's inside?"
    return


### REACTIONS [RCT100]

init 5 python:
    addReaction("mas_reaction_generic", None)

label mas_reaction_generic:
    "This is a test"
    return

#init 5 python:
#    addReaction("mas_reaction_gift_generic", None)

label mas_reaction_gift_generic:
    if random.randint(1,2) == 1:
        m 1esd "[player], are you trying to give me something?"
        m 1rssdlb "I found it, but I can’t bring it here..."
        m "I can’t seem to read it well enough."
        m 3esa "But that’s alright!"
        m 1esa "It’s the thought that counts after all, right?"
        m "Thanks for being so thoughtful, [player]~"
    else:
        m 2dkd "{i}*sigh*{/i}"
        m 4ekc "I’m sorry, [player]."
        m 1ekd "I know you’re trying to give me something."
        m 2rksdld "But for some reason I can’t read the file."
        m 3euc "Don’t get me wrong, however."
        m 3eka "I still appreciate that you tried giving something to me."
        m 1hub "And for that, I’m thankful~"
    $ store.mas_filereacts.delete_file(None)
    return

#init 5 python:
#    addReaction("mas_reaction_gift_test1", "test1")

label mas_reaction_gift_test1:
    m "Thank you for gift test 1!"

    $ gift_ev = mas_getEV("mas_reaction_gift_test1")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    return

#init 5 python:
#    addReaction("mas_reaction_gift_test2", "test2")

label mas_reaction_gift_test2:
    m "Thank you for gift test 2!"

    $ gift_ev = mas_getEV("mas_reaction_gift_test2")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    return

## coffee vars
# NOTE: this is just for reference, check sprite-chart for inits
# persistent._mas_acs_enable_coffee
# persistent._mas_coffee_brewing

init 5 python:
    addReaction("mas_reaction_gift_coffee", "coffee", is_good=True)

label mas_reaction_gift_coffee:

    m 1euc "Hmm?"
    $ store.mas_sprites.reset_zoom()
    m 1euc "Oh,{w} is this coffee?"
    $ mas_receivedGift("mas_reaction_gift_coffee")

    if persistent._mas_coffee_been_given:
        $ mas_gainAffection(bypass=True)
        m 1wuo "It's a flavor I've haven't had before, too."
        m 1hua "I can't wait to try it!"
        m "Thank you so much, [player]!"

    else:
        show emptydesk at i11 zorder 9
        $ mas_gainAffection(modifier=2, bypass=True)

        m 1hua "Now I can finally make some!"
        m "Thank you so much, [player]!"
        m "Why don't I go ahead and make a cup right now?"
        m 1eua "I'd like to share the first with you, after all."

        # monika is off screen
        hide monika with dissolve
        pause 2.0
        m "I know there's a coffee machine somewhere around here...{w=2}{nw}"
        m "Ah, there it is!{w=2}{nw}"
        pause 5.0
        m "And there we go!{w=2}{nw}"
        show monika 1eua at i11 zorder MAS_MONIKA_Z with dissolve
        hide emptydesk

        # monika back on screen
        m 1eua "I'll let that brew for a few minutes."
        $ mas_brewCoffee()
        $ persistent._mas_acs_enable_coffee = True
        $ persistent._mas_coffee_been_given = True

    $ gift_ev = mas_getEV("mas_reaction_gift_coffee")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    return

init 5 python:
    addReaction("mas_reaction_quetzal_plush", "quetzalplushie", is_good=True)

label mas_reaction_quetzal_plush:
    if not persistent._mas_acs_enable_quetzalplushie:
        $ mas_receivedGift("mas_reaction_quetzal_plush")
        $ mas_gainAffection(modifier=2, bypass=True)
        m 1wud "Oh!"
        $ monika_chr.wear_acs_pst(mas_acs_quetzalplushie)
        $ persistent._mas_acs_enable_quetzalplushie = True
        m 1sub "It’s a quetzal!"
        m "Oh my gosh, thanks a lot, [player]!"
        m 1eua "I did mention that I’d like to have a quetzal as a pet..."
        m 1rud "But I would never force the poor thing to stay."
        m 1hua "And now you gave me the next closest thing!"
        m 1hub "This makes me so happy!"
        if mas_isMoniAff(higher=True):
            m 5esbfa "You always seem to know how to make me smile."

        m 1hsb "Thank you again, [player]~"
    else:
        m 1rksdlb "You already gave me a quetzal plushie, [player]."
    $ gift_ev = mas_getEV("mas_reaction_quetzal_plush")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    return

#This one is added later so the init pipeline can define the anni function
init 11 python:
    # only available after 6 months or if it's her birthday, may as well add valentine later
    # TODO add dialogue for gift rejection in case the conditions below are not met 
    if mas_anni.pastSixMonths() or mas_isMonikaBirthday():
        addReaction("mas_reaction_promisering", "promisering", is_good=True)

label mas_reaction_promisering:
    if not persistent._mas_acs_enable_promisering:
        $ mas_receivedGift("mas_reaction_promisering")
        if mas_isMoniEnamored(higher=True):
            $ mas_gainAffection(modifier=5, bypass=True)
            $ monika_chr.wear_acs_pst(mas_acs_promisering)
            $ persistent._mas_acs_enable_promisering = True
            m 1wud "Is that... a..."
            m "..."
            m 1wka "I..."
            m 1wkbltpa "I'm sorry, [player], I just..."
            m 1dkbltpa "You might not have even meant much by it, but..."
            m "Just in case this is a promise from you..."
            m 3lkbltpa "Know that I’ll cherish it."
            m 3dkbltpa "Always."
            m 1skbltpa "This makes me so happy!"
            if mas_isSpecialDay():
                #TODO maybe go more in detail for this
                m "Even more that you gave it to me on this special day ..."
            m 1dkbltpa "Aha, sorry for crying, [player]..."
            m 1skbla "I’m just really, really happy right now."
            m 1dkbla "Thank you."
            m "I love you, [player]."
            m "More than anything else in this fleeting world."
        elif mas_isMoniNormal(higher=True):
            $ persistent._mas_acs_enable_promisering = True
            $ mas_gainAffection(modifier=3, bypass=True)
            $ monika_chr.wear_acs_pst(mas_acs_promisering)
            m 1wud "Oh... What is this, [player]?"
            m "Is that... a ring?"
            m 1sub " I can’t believe it, [player]!"
            m 1hua "This...{w} means a lot to me."
            m 3hub "This makes me so happy!"
            m 3eub "Seriously, thank you so much for this, [player]!"
        else:
            $ mas_gainAffection(bypass=True)
            m 1wud "Is that a ring?"
            m "That's very ..."
            m "Unexpected."
            if mas_isMoniDis(lower=True):
                m 2rkc "I appreciate the thought ...{w} but I can't accept it."
                m 2ekc "Sorry, [player]."
                $ persistent._mas_acs_enable_promisering = False
            else:
                $ monika_chr.wear_acs_pst(mas_acs_promisering)
                $ persistent._mas_acs_enable_promisering = True
                m 3hua "I'm happily surprised by this, [player]."
                m "Thanks."
    else:
        m 1rksdlb "[player]..."
        m 1rusdlb "You already gave me a ring!"
    $ gift_ev = mas_getEV("mas_reaction_promisering")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    return

init 5 python:
    addReaction("mas_reaction_plush", "plushie", is_good=True)

label mas_reaction_plush:
    m 1wud "What’s this, [player]?"
    m "Are you trying to give me a plushie?"
    m 1rksdlb "I appreciate the thought, but ..."
    m 1ekd "For some reason, I can’t seem to bring it here."
    m 1rkc "I wish I could ..."
    m 1hua "But don’t worry, [player]!"
    m 1hub "Ehehe~"
    m 1hua "Thank you for trying!"
    $ mas_receivedGift("mas_reaction_plush") # while unsuccessful counts
    $ gift_ev = mas_getEV("mas_reaction_plush")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    return

init 5 python:
    addReaction("mas_reaction_bday_cake", "birthdaycake")

label mas_reaction_bday_cake:
    if not mas_isMonikaBirthday():
        $ mas_loseAffection(3)
        m 1rksdlb "Today is not my birthday, did you forget when it is, [player]?"
    else:
        $ mas_gainAffection(modifier=2, bypass=True)
        m 1wud "Oh, [player]..."
        m "A... birthday cake?"
        m 3sub "Gosh, this is so sweet of you!"
        m 1hub "I really love this a lot!"
        m 1dkbfa "I’ll save it for now"
        m 3hua "I can’t wait to try it later!"
        m 1dsc "But..."
        m 3ekbfa "I wish we could eat it together..."
        m 1dkbfa "A birthday cake is for sharing, after all~"
        m 1ekbfa "Thank you for this, [player]."
        $ mas_receivedGift("mas_reaction_bday_cake")
        if mas_isMoniAff(higher=True):
            m 3hubfb "I love you! Ehehe~"
    $ gift_ev = mas_getEV("mas_reaction_bday_cake")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    return

init 5 python:
    addReaction("mas_reaction_cupcake", "cupcake", is_good=False)

label mas_reaction_cupcake:
    m 1wud "Is that a...cupcake?"
    m 3hub "Wow, thanks [player]!"
    m 3euc "Come to think of it, I’ve been meaning to make some cupcakes myself."
    m 1eua "I wanted to learn how to bake good pastries like Natsuki did."
    m 1rksdlb "Buuut I’ve yet to make a kitchen to use!"
    m 3eub "Maybe in the future once I get better at programming, I’ll be able to make one here."
    m 5hubfa "Would be nice to have another hobby other than writing, ehehe~"
    $ mas_receivedGift("mas_reaction_cupcake")
    $ gift_ev = mas_getEV("mas_reaction_cupcake")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    return

init 5 python:
    addReaction("mas_reaction_knife", "knife", is_good=False)

label mas_reaction_knife:
    m 1euc "...?"
    m 1wud "Is that...a knife?"
    m 2wfc "Why would you want to give me that?"
    m 2wfd "I don’t need this here!."
    m 3tfc "...Someone else, maybe."
    m 1dfc "..."
    m 1rsc "I’m not taking this, [player]."
    m 1rfc "If you were trying to be funny, then you have {i}very{/i} poor taste."
    $ mas_receivedGift("mas_reaction_knife") # while technically she didn't accept this one counts
    $ gift_ev = mas_getEV("mas_reaction_knife")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    return

# ending label for gift reactions, this just resets a thing
label mas_reaction_end:
    $ persistent._mas_filereacts_just_reacted = False
    return

init 5 python:
    # TODO ideally we should comment on this gift in any date
    # so it requires special dialogue, until we have that let's keep it O31 only
    if mas_isO31():
        addReaction("mas_reaction_candy", "candy", is_good=True)

label mas_reaction_candy:
    $ times_candy_given = mas_getGiftStatsForDate("mas_reaction_candy")
    if times_candy_given == 0:
        $ mas_gainAffection(10,bypass=True)
        m 1wua "Oh...{w=1}what's this?"
        m 1sua "You got me candy, [player], yay!"
        m 1eka "That's so {i}sweet{/i}..."
        m 1hub "Ahaha!"
        m 1eka "Kidding aside, that's really nice of you."
        m 2lksdlc "I don't get to have much candy anymore, and it just wouldn't be Halloween without it..."
        m 1eka "So thank you, [player]..."
        m 1eka "You always know exactly what will make me happy~"
        m 1hub "Now let's enjoy some of this delicious candy!"
    elif times_candy_given == 1:
        $ mas_gainAffection(5,bypass=True)
        m 1wua "Aww, you got me more candy, [player]?"
        m 1hub "Thank you!"
        m 3tku "The first batch was {i}sooo{/i} good, I couldn't wait to have more."
        m 1hua "You really do spoil me, [player]~"
    elif times_candy_given == 2:
        $ mas_gainAffection(3,bypass=True)
        m 1wud "Wow, even {i}more{/i} candy, [player]?"
        m 1eka "That's really nice of you..."
        m 1lksdla "But I think this is enough."
        m 1lksdlb "I'm already feeling jittery from all the sugar, ahaha!"
        m 1ekbfa "The only sweetness I need now is you~"
    elif times_candy_given == 3:
        m 2wud "[player]...{w=1} you got me {i}even more{/i} candy?!"
        m 2lksdla "I really do appreciate it, but I told you I've had enough for one day..."
        m 2lksdlb "If I eat anymore I'm going to get sick, ahaha!"
        m 1eka "And you wouldn't want that, right?"
    elif times_candy_given == 4:
        $ mas_loseAffection(5)
        m 2wfd "[player]!"
        m 2tfd "Are you not listening to me?"
        m 2tfc "I told you I don't want anymore candy today!"
        m 2ekc "So please, stop."
        m 2rkc "It was really nice of you to get me all of this candy on Halloween, but enough is enough..."
        m 2ekc "I can't eat all of this."
    else:
        $ mas_loseAffection(10)
        m 2tfc "..."
        python:
            store.mas_ptod.rst_cn()
            local_ctx = {
                "basedir": renpy.config.basedir
            }
        show monika at t22
        show screen mas_py_console_teaching

        call mas_wx_cmd("import os", local_ctx, w_wait=1.0)
        call mas_wx_cmd("os.remove(os.path.normcase(basedir+'/characters/candy.gift'))", local_ctx, w_wait=1.0, x_wait=1.0)
        $ store.mas_ptod.ex_cn()
        hide screen mas_py_console_teaching
        show monika at t11

    $ mas_receivedGift("mas_reaction_candy")
    $ gift_ev = mas_getEV("mas_reaction_candy")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    $ persistent._mas_filereacts_reacted_map.pop(gift_ev.category,None)
    return

init 5 python:
    # TODO ideally we should comment on this gift in any date
    # so it requires special dialogue, until we have that let's keep it O31 only
    if mas_isO31():
        addReaction("mas_reaction_candycorn", "candycorn", is_good=False)

label mas_reaction_candycorn:
    $ times_candy_given = mas_getGiftStatsForDate("mas_reaction_candycorn")
    if times_candy_given == 0:
        $ mas_gainAffection(3,bypass=True)
        m 1wua "Oh...{w=1}what's this?"
        m 1eka "Aww did you get me candy, [player]?"
        m 1hua "Yay!"
        m 3eub "Let's see what you got for me..."
        m 4ekc "..."
        m 2eka "Oh...{w=2}candy corn."
        m 2eka "..."
        m 2lksdla "That's really nice of you..."
        m 2lksdla "But...{w=1}umm...{w=1}I don't actually like candy corn."
        m 2hksdlb "Sorry, ahaha..."
        m 4eka "I do appreciate you trying to give me candy on Halloween, though."
        m 1hua "And if you could find a way to get some other candy for me, it'd make me really happy, [player]!"
    elif times_candy_given == 1:
        $ mas_loseAffection(5)
        m 2esc "Oh."
        m 2esc "More candy corn, [player]?"
        m 4esc "I already told you I don't really like candy corn."
        m 4ekc "So could you please try to find something else?"
        m 1eka "I don't get sweets that often anymore..."
        m 1ekbfa "Well...{w=1}besides you, [player]..."
        m 1hubfa "Ehehe~"
    elif times_candy_given == 2:
        $ mas_loseAffection(10)
        m 2wfw "[player]!"
        m 2tfc "I really tried not to be rude about this, but..."
        m 2tfc "I keep telling you I don't like candy corn and you just keep giving it to me anyways."
        m 2rfc "It's starting to feel like you're just trying to mess with me at this point."
        m 2tkc "So please, either find me some other kind of candy or just stop."
    else:
        $ mas_loseAffection(15) # should have seen it coming
        m 2tfc "..."
        python:
            store.mas_ptod.rst_cn()
            local_ctx = {
                "basedir": renpy.config.basedir
            }
        show monika at t22
        show screen mas_py_console_teaching

        call mas_wx_cmd("import os", local_ctx, w_wait=1.0)
        call mas_wx_cmd("os.remove(os.path.normcase(basedir+'/characters/candycorn.gift'))", local_ctx, w_wait=1.0, x_wait=1.0)
        $ store.mas_ptod.ex_cn()
        hide screen mas_py_console_teaching
        show monika at t11

    $ mas_receivedGift("mas_reaction_candycorn") # while technically she didn't accept this one counts
    $ gift_ev = mas_getEV("mas_reaction_candycorn")
    $ store.mas_filereacts.delete_file(gift_ev.category)
    #TODO check if this is a good way to allow multi gifts
    $ persistent._mas_filereacts_reacted_map.pop(gift_ev.category,None)
    return
