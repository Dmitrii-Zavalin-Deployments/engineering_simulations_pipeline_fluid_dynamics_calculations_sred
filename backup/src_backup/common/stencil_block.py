# src/common/stencil_block.py

import logging

from src.common.base_container import ValidatedContainer

from .cell import Cell

logger = logging.getLogger(__name__)

class StencilBlock(ValidatedContainer):
    """
    Logical Wiring: Represents the 7-point stencil topology.
    Acts as the graph node connecting neighboring Cells.
    """

    __slots__ = [
        '_center',
        '_dt',
        '_dx',
        '_dy',
        '_dz',
        '_f_vals',
        '_i_minus',
        '_i_plus',
        '_id',
        '_j_minus',
        '_j_plus',
        '_k_minus',
        '_k_plus',
        '_mu',
        '_rho'
    ]

    def __init__(self, center: Cell, i_minus: Cell, i_plus: Cell, 
                 j_minus: Cell, j_plus: Cell, k_minus: Cell, k_plus: Cell,
                 dx: float, dy: float, dz: float, dt: float, 
                 rho: float, mu: float, f_vals: tuple):
        
        # Unique ID for tracking in logs - assigned first for traceability
        object.__setattr__(self, '_id', f"Block_{id(self) % 1000}")
        
        # Explicit initialization (Rule 5: Deterministic Initialization)
        object.__setattr__(self, '_center', center)
        object.__setattr__(self, '_i_minus', i_minus)
        object.__setattr__(self, '_i_plus', i_plus)
        object.__setattr__(self, '_j_minus', j_minus)
        object.__setattr__(self, '_j_plus', j_plus)
        object.__setattr__(self, '_k_minus', k_minus)
        object.__setattr__(self, '_k_plus', k_plus)
        
        # Physics attributes (Cached at assembly for performance)
        object.__setattr__(self, '_dx', float(dx))
        object.__setattr__(self, '_dy', float(dy))
        object.__setattr__(self, '_dz', float(dz))
        object.__setattr__(self, '_dt', float(dt))
        object.__setattr__(self, '_rho', float(rho))
        object.__setattr__(self, '_mu', float(mu))
        object.__setattr__(self, '_f_vals', tuple(f_vals))

    # --- Topological Accessors ---
    @property
    def id(self) -> str: return self._id

    @property
    def center(self) -> Cell: return self._center
    @property
    def i_minus(self) -> Cell: return self._i_minus
    @property
    def i_plus(self) -> Cell: return self._i_plus
    @property
    def j_minus(self) -> Cell: return self._j_minus
    @property
    def j_plus(self) -> Cell: return self._j_plus
    @property
    def k_minus(self) -> Cell: return self._k_minus
    @property
    def k_plus(self) -> Cell: return self._k_plus

    @property
    def dx(self) -> float: return self._dx
    @property
    def dy(self) -> float: return self._dy
    @property
    def dz(self) -> float: return self._dz
    
    @property
    def dt(self) -> float: 
        return self._dt

    @dt.setter
    def dt(self, value: float):
        """
        Rule 4 Sync Gate: Allows ElasticityManager to update the block's time-step.
        """
        if value <= 0:
            logger.error(f"STABILITY CRASH: {self._id} rejected invalid dt={value}")
            raise ValueError(f"Numerical Instability: dt must be positive, got {value}")
        
        # --- [STRATEGIC DIAGNOSTIC LOG] ---
        logger.debug(f"SYNC [Physics]: {self._id} updated dt -> {value:.4e}")
        
        object.__setattr__(self, '_dt', float(value))
    
    @property
    def rho(self) -> float: return self._rho
    @property
    def mu(self) -> float: return self._mu
    @property
    def f_vals(self) -> tuple: return self._f_vals

    def __repr__(self):
        return f"<{self._id} dt={self._dt:.2e}>"