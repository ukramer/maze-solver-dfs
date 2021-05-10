import sys
from enum import Enum
from typing import Dict, Type, TypeVar, Tuple, List, Optional

import numpy as np
from numpy import ndarray

M = TypeVar('M', bound='Maze')


class Movement(Enum):
    NORTH: str = 'N'
    EAST: str = 'E'
    SOUTH: str = 'S'
    WEST: str = 'W'

    @staticmethod
    def get_opposite(movement: "Movement") -> "Movement":
        if movement == Movement.NORTH:
            return Movement.SOUTH
        if movement == Movement.EAST:
            return Movement.WEST
        if movement == Movement.SOUTH:
            return Movement.NORTH
        return Movement.EAST


class Maze(object):
    # Possible Movements (North, East, South, West)
    MOVEMENTS: List[Movement] = [Movement.NORTH, Movement.EAST,
                                 Movement.SOUTH, Movement.WEST]

    # Some constants for the replacement with the chars in the maze file
    START_POINT: int = 2
    GOAL_POINT: int = 3
    EMPTY: int = 1
    WALL: int = 0
    NORTH: int = 4
    EAST: int = 5
    SOUTH: int = 6
    WEST: int = 7
    CHAR_MAP: Dict[str, int] = {
        "*": WALL,
        " ": EMPTY,
        "A": START_POINT,
        "B": GOAL_POINT,
        "N": NORTH,
        "E": EAST,
        "S": SOUTH,
        "W": WEST
    }

    # Container for the two-dimensional list
    maze: ndarray

    # Points in the maze representing current position and position of
    # goal
    current_pos: Tuple[int, int] = (0, 0)
    goal_pos: Tuple[int, int] = (0, 0)

    # Store of the path which has been followed
    path: List[Tuple[int, int]] = []
    # Store of all the movements which have been made
    # Those movements are in relation to the points in the path List
    movements: List[Movement] = []

    def __init__(self, maze: ndarray) -> None:
        self.maze = maze

    def get_start_point(self) -> Tuple[int, int]:
        return self.__get_point(self.START_POINT)

    def get_goal_point(self) -> Tuple[int, int]:
        return self.__get_point(self.GOAL_POINT)

    def solve(self) -> bool:
        self.current_pos = self.get_start_point()
        self.goal_pos = self.get_goal_point()

        # loop as long as we don't reach goal position
        while True:
            # get the next possible movement
            next_movement = self.__get_next_possible_movement()
            while next_movement is None:
                # no possible movement anymore, we need to step back
                # get back as soon as we have another possible movement
                self.path.pop()
                if len(self.path) == 0:
                    # no solution found as we are at the beginning again
                    return False

                self.current_pos = self.path[-1]
                last_movement = self.movements.pop()
                next_movement = self.__get_next_possible_movement(
                    last_movement)

            # store the next movement as movement, current position and
            # path
            self.current_pos = \
                self.__get_point_by_movement(
                    self.current_pos, next_movement
                )
            if self.current_pos == self.goal_pos:
                return True

            self.movements.append(next_movement)
            self.path.append(self.current_pos)

    @classmethod
    def from_txt(cls: Type[M], file: str) -> M:
        # As per the suggestion on
        # https://stackoverflow.com/questions/19792112/static-method-returning-an-instance-of-its-class
        # I needed to put Maze into quotes as it is not recognized here
        try:
            f = open(file, "r")
        except FileNotFoundError:
            print("Could not open file {}".format(file),
                  file=sys.stderr)
            exit(1)

        lines = f.read().splitlines()
        f.close()

        matrix = []
        for line in lines:
            for i, j in Maze.CHAR_MAP.items():
                line = line.replace(i, str(j))
            elements = list(map(int, list(line)))
            matrix.append(np.array(elements))
        maze = np.array(matrix)
        return cls(maze)

    @staticmethod
    def get_maze(maze: ndarray) -> str:
        output = ''
        for line in maze:
            line = list(map(str, list(line)))
            str_line = ''.join(line)
            for i, j in Maze.CHAR_MAP.items():
                str_line = str_line.replace(str(j), i)
            output += str_line + "\n"
        return output

    def get_maze_with_movements(self) -> str:
        maze = self.maze.copy()
        movements = self.movements.copy()
        for point in self.path:
            x, y = point
            maze[y][x] = self.CHAR_MAP[movements.pop(0).value]
        return self.get_maze(maze)

    def get_path(self) -> str:
        return ''.join([movement.value for movement in self.movements])

    def __str__(self) -> str:
        return self.get_maze(self.maze)

    @staticmethod
    def __get_point_by_movement(point: Tuple[int, int],
                                movement: Movement) -> Tuple[int, int]:
        x, y = point
        if movement == Movement.NORTH:
            return x, y - 1
        if movement == Movement.EAST:
            return x + 1, y
        if movement == Movement.SOUTH:
            return x, y + 1
        if movement == Movement.WEST:
            return x - 1, y
        raise Exception("Unknown movement {}".format(movement))

    def __get_point(self, point: int) \
            -> Tuple[int, int]:
        # Search for start or goal point
        result = np.where(self.maze == point)
        y, x = int(result[0]), int(result[1])
        return x, y

    def __is_valid_point(self, point: Tuple[int, int]) -> bool:
        x, y = point
        if point in self.path:
            # Detect cycle when the next point is already in the path
            return False
        return self.maze[y, x] and \
               self.maze[y, x] in [self.EMPTY, self.GOAL_POINT]

    def __get_next_possible_movement(
            self, already_tried_movement: Optional[Movement] = None
    ) -> Optional[Movement]:
        movements = self.MOVEMENTS
        if already_tried_movement:
            index_last_movement = self.MOVEMENTS.index(
                already_tried_movement)
            movements = self.MOVEMENTS[index_last_movement + 1:]

        for movement in movements:
            if len(self.movements) > 0:
                # get the movement which would be the opposite of the
                # last movement the opposite is not allowed as it would
                # be an infinite loop
                movement_back = Movement.get_opposite(self.movements[-1])
                if movement == movement_back:
                    continue

            next_point = self.__get_point_by_movement(
                self.current_pos, movement
            )
            if self.__is_valid_point(next_point):
                return movement
        return None


def main(file: str) -> None:
    # initialize Maze object by text file
    maze = Maze.from_txt(file)
    # start to find a path through the maze
    if maze.solve():
        print('Unsolved maze:')
        print(maze)

        print('Solve maze:')
        print(maze.get_maze_with_movements())

        print('Path: {}'.format(maze.get_path()))
    else:
        print("No solution could be found")


if __name__ == "__main__":
    # check whether argument is set
    if sys.argv[1] is None:
        print("Missing file parameter", file=sys.stderr)
        exit(1)

    main(sys.argv[1])
