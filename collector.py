import abc
import prometheus_client
import time
import yaml
import zfslib as zfs

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from prometheus_client import start_http_server, Metric
from prometheus_client.core import REGISTRY, GaugeMetricFamily, InfoMetricFamily
from prometheus_client.registry import Collector
from typing import Iterable, override, Optional, Any, SupportsFloat

MetricFamilies = type[GaugeMetricFamily | InfoMetricFamily]
ZfsDataset = type[zfs.Dataset | zfs.Pool]


class MetricHelper[M: MetricFamilies, V](abc.ABC):
    LabelAccessor = Callable[[zfs.Pool | zfs.Dataset], str] | str
    ResultTransformer = Optional[Callable[[Any], V]]

    metric: MetricFamilies

    def get_label_value_safe(self, entity: zfs.Pool | zfs.Dataset, accessor: LabelAccessor) -> str:
        if isinstance(accessor, str):
            return accessor
        value = accessor(entity)
        if value is None:
            return '<null>'
        return str(value)

    def new_metric(self) -> None:
        self.metric = self.metric_class(name=self.metric_name, documentation=self.doc,
                                        labels=self.labels_to_getters.keys())

    def __init__(self, metric_class: type[M], kind: str, prop: str,
                 labels_to_getters: OrderedDict[str, LabelAccessor], result_transformer: ResultTransformer):
        self.metric_name = f'zfs_{kind}_{prop}'
        self.target_property = prop
        self.labels_to_getters = labels_to_getters
        # TODO make docs not useless
        self.doc = f'{kind} property {prop}'
        self.metric_class = metric_class
        self.result_transformer = result_transformer

    @abstractmethod
    def final_metric_transform(self, value: Any): ...

    def add_to_metric(self, entity: zfs.Pool | zfs.Dataset) -> None:
        try:
            value = entity.get_property(self.target_property)
        except KeyError:
            # TODO proper logging
            print(f'Unable to find "{self.target_property}" on {entity}')
            return
        if self.result_transformer is not None:
            value = self.result_transformer(value)
        value = self.final_metric_transform(value)
        label_values = [
            self.get_label_value_safe(entity, label_accessor) for label_accessor in self.labels_to_getters.values()
        ]
        self.metric.add_metric(labels=label_values, value=value)


class GaugeMetricHelper(MetricHelper[GaugeMetricFamily, float]):
    ResultTransformer = Optional[Callable[[Any], float]]

    def __init__(self, kind, prop, labels_to_getters, result_transformer: ResultTransformer):
        super().__init__(GaugeMetricFamily, kind, prop, labels_to_getters, result_transformer)

    @override
    def final_metric_transform(self, value: Any):
        return float(value)


class InfoMetricHelper(MetricHelper[InfoMetricFamily, str]):
    ResultTransformer = Optional[Callable[[Any], str]]

    def __init__(self, kind, prop, labels_to_getters, result_transformer: ResultTransformer):
        super().__init__(InfoMetricFamily, kind, prop, labels_to_getters, result_transformer)

    @override
    def final_metric_transform(self, value: Any):
        return {self.target_property: str(value)}


class MetricMaker[Z](abc.ABC):
    props: list[str]
    metric_helpers: list[MetricHelper]
    labels_to_getters: OrderedDict[str, Callable[[Z], str] | str]

    def __init__(self, extended_properties: bool):
        self._extended_properties = extended_properties

    def get_props(self, config: dict) -> list[str]:
        props = list(config['default'])
        if self._extended_properties:
            props.extend(config['extended'])
        return props

    def create_metrics[V, H: MetricHelper](self, helper_class: type[H], props_config: dict, kind: str,
                                           result_transformer: MetricHelper[..., V].ResultTransformer = None
                                           ) -> dict[str, H]:
        props = self.get_props(props_config)
        props_to_metrics = {}
        for prop in props:
            props_to_metrics[prop] = helper_class(kind, prop, self.labels_to_getters, result_transformer)
        return props_to_metrics

    def get_zfs_query_props(self) -> list[str]:
        return list(self.props)

    def add_to_metrics(self, entity: Z) -> None:
        for metric_helper in self.metric_helpers:
            metric_helper.add_to_metric(entity)

    @property
    def metrics(self) -> list[MetricFamilies]:
        return [helper.metric for helper in self.metric_helpers]

    def reset_metrics(self) -> None:
        for metric_helper in self.metric_helpers:
            metric_helper.new_metric()


class ZpoolMetricMaker(MetricMaker[zfs.Pool]):
    def __init__(self, props_config: dict, extended_properties: bool):
        super().__init__(extended_properties)

        self.labels_to_getters = OrderedDict({
            'pool': lambda p: p.name,
        })

        props_to_metrics = dict(self.create_metrics(GaugeMetricHelper, props_config['gauges'], 'pool'))
        props_to_metrics.update(self.create_metrics(InfoMetricHelper, props_config['info'], 'pool'))
        self.props = list(props_to_metrics.keys())
        self.metric_helpers = list(props_to_metrics.values())


class DatasetMetricMaker(MetricMaker[ZfsDataset]):
    @staticmethod
    def none_to_float(value: Any) -> float:
        if isinstance(value, SupportsFloat):
            return float(value)
        if isinstance(value, str):
            if 'none' == value or '-' == value:
                return -1.0
            raise f'Unexpected string value in none_to_float: "{value}"'
        if value is None:
            return -1.0
        raise f'Not implemented: {value.__class__}  value: {value}'

    @staticmethod
    def create_label_accessor(label_property):
        def accessor(entity):
            return entity.get_property(label_property)
        return accessor

    def __init__(self, props_config: dict, extended_properties: bool):
        super().__init__(extended_properties)

        self._label_props = props_config['labels']
        self.labels_to_getters = OrderedDict({
            'name': lambda d: d.path,
            'pool': lambda d: d.pool.name,
        })
        for label_prop in self._label_props:
            self.labels_to_getters[label_prop] = self.create_label_accessor(label_prop)

        props_to_metrics = dict(self.create_metrics(GaugeMetricHelper, props_config['gauges'], 'dataset'))
        props_to_metrics.update(self.create_metrics(GaugeMetricHelper, props_config['gauges_if_not_none'],
                                                    'dataset', DatasetMetricMaker.none_to_float))
        props_to_metrics.update(self.create_metrics(InfoMetricHelper, props_config['info'], 'dataset'))
        props_to_metrics.update(self.create_metrics(InfoMetricHelper, props_config['info_date'], 'dataset', str))
        self.props = list(props_to_metrics.keys())
        self.props.extend(self._label_props)
        self.metric_helpers = list(props_to_metrics.values())


class ZfsCollector(Collector):
    def __init__(self, extended_properties: bool):
        source_dir = Path(__file__).parent
        with (source_dir / 'config.yaml').open('r') as f:
            self.config = yaml.safe_load(f)

        self._zpool_metric_maker = ZpoolMetricMaker(self.config['properties']['zpool'], extended_properties)
        self._dataset_metric_maker = DatasetMetricMaker(self.config['properties']['zfs'], extended_properties)

        self._zfs_conn = zfs.Connection()

    def _get_zpools_and_datasets(self) -> tuple[list[zfs.Pool], list[ZfsDataset]]:
        pools = list(self._zfs_conn.load_poolset(
            zfs_props=self._dataset_metric_maker.get_zfs_query_props(),
            zpool_props=self._zpool_metric_maker.get_zfs_query_props(),
            get_mounts=True
        ).items)

        datasets = list(pools) # library doesn't include the root dataset in get_all_datasets()
        for pool in pools:
            datasets.extend(pool.get_all_datasets())
        return pools, datasets

    def collect(self) -> Iterable[Metric]:
        self._zpool_metric_maker.reset_metrics()
        self._dataset_metric_maker.reset_metrics()

        pools, datasets = self._get_zpools_and_datasets()
        for pool in pools:
            self._zpool_metric_maker.add_to_metrics(pool)
        for dataset in datasets:
            self._dataset_metric_maker.add_to_metrics(dataset)

        yield from self._zpool_metric_maker.metrics
        yield from self._dataset_metric_maker.metrics


if __name__ == '__main__':
    # TODO make this optional script parameter
    REGISTRY.unregister(prometheus_client.GC_COLLECTOR)
    REGISTRY.unregister(prometheus_client.PLATFORM_COLLECTOR)
    REGISTRY.unregister(prometheus_client.PROCESS_COLLECTOR)

    # TODO extended_properties as script parameter
    REGISTRY.register(ZfsCollector(extended_properties=False))

    start_http_server(9126)

    while True:
        time.sleep(3600)
