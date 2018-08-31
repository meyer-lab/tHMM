def FlagsForFile(*args, **kwargs):
    return {
        'flags': [
            '--std=c++17',
            '-Wall',
            '-I/usr/include/cppunit/',
            '-lstdc++',
            '-lcppunit'
        ],
    }